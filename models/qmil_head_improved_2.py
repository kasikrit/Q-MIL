"""
Quantum-Inspired Multiple Instance Learning (Q-MIL) Head and Utilities
======================================================================
Provides the core PyTorch architecture for the Q-MIL framework, integrating 
quantum state projection, Hamiltonian attention, and density matrix formulation 
for robust hematopathology triage (IDA vs. THL). 

Also includes dataset utilities, TensorFlow-based feature extraction pipelines, 
and dataloader construction methods.
"""

import os
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import tensorflow as tf
from PatchDatasetPreserve import PatchDatasetPreserve


class QMILHead(nn.Module):
    """
    Quantum-Inspired Multiple Instance Learning (Q-MIL) Head.

    Args:
        d_in (int): Feature dimension from the backbone (e.g., 1536 for ConvNeXt-Large).
        d_model (int): Internal latent dimension for the quantum state space.
        num_classes (int): Number of diagnostic classes (default: 2 for IDA vs. THL).
        d_k (int): Key/Query projection dimension for the Hamiltonian attention.
        eps (float): Numerical stability floor to prevent division by zero.
    """

    def __init__(self, d_in: int = 1536, d_model: int = 256,
                 num_classes: int = 2, d_k: int = 64, eps: float = 1e-9):
        super(QMILHead, self).__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.d_k = d_k
        self.eps = eps

        # 1. Quantum State Projection
        # Projects backbone feature dimension to latent quantum dimension and stabilizes.
        self.patch_proj = nn.Sequential(
            nn.Linear(d_in, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # 2. Learnable Purity Gate
        # Maps projected patch embeddings to a scalar alpha_i in (0, 1), 
        # downweighting noisy/artifact patches to preserve magnitude signals.
        self.purity_gate = nn.Sequential(
            nn.Linear(d_model, 1, bias=True),
            nn.Sigmoid(),
        )

        # 3. Entanglement Coupling (Hamiltonian / Self-Attention)
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)

        # 4. Quantum Observables (Hermitian Operators)
        self.raw_observables = nn.Parameter(
            torch.Tensor(num_classes, d_model, d_model)
        )
        nn.init.xavier_uniform_(self.raw_observables)

    def get_symmetric_observables(self) -> torch.Tensor:
        """Enforces Hermitian (symmetric) observables: O_sym = 0.5*(O + O^T)."""
        return 0.5 * (self.raw_observables + self.raw_observables.transpose(1, 2))

    def forward(self, z_raw: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass for Q-MIL Head.

        Args:
            z_raw (torch.Tensor): Raw patch features from backbone. Shape (B, N, d_in).
            mask (torch.Tensor): Boolean mask where True = valid patch. Shape (B, N).

        Returns:
            logits (torch.Tensor): Diagnostic logits. Shape (B, num_classes).
            entropy (torch.Tensor): von Neumann entropy per patient bag. Shape (B,).
            rho_bag (torch.Tensor): Bag density matrices. Shape (B, d_model, d_model).
            alpha (torch.Tensor): Per-patch purity weights. Shape (B, N).
        """
        B, N, _ = z_raw.shape

        # Step 1: Quantum State Projection
        z_proj = self.patch_proj(z_raw)  # (B, N, d_model)

        # Step 2: Bag-Level Centering
        if mask is not None:
            valid_counts = mask.sum(dim=1, keepdim=True).unsqueeze(-1)
            z_sum = (z_proj * mask.unsqueeze(-1)).sum(dim=1, keepdim=True)
            z_mean = z_sum / (valid_counts + self.eps)
            z_proj = torch.where(mask.unsqueeze(-1), z_proj - z_mean, torch.zeros_like(z_proj))
        else:
            z_proj = z_proj - z_proj.mean(dim=1, keepdim=True)

        # Step 3: Compute Purity Weights and L2 Normalization
        alpha = self.purity_gate(z_proj).squeeze(-1)  # (B, N)
        z = F.normalize(z_proj, p=2, dim=-1, eps=self.eps)
        z = z * alpha.unsqueeze(-1)  # Weighted ket: α_i |ψ_i⟩

        # Step 4: Entanglement Coupling Generation
        Q = self.W_q(z)
        K = self.W_k(z)
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.d_k ** 0.5)

        if mask is not None:
            attn_mask = mask.unsqueeze(1).expand(-1, N, -1)
            attention_scores = attention_scores.masked_fill(~attn_mask, float('-inf'))

        W = F.softmax(attention_scores, dim=-1)

        if mask is not None:
            W = W * mask.unsqueeze(-1)

        # Step 5: Bag-Level Density Matrix Construction
        # Residual connection retains individual morphological variance
        z_entangled = z + torch.bmm(W, z)
        rho_tilde = torch.bmm(z_entangled.transpose(1, 2), z_entangled)
        
        # Trace normalization ensuring Tr(ρ) = 1
        traces = torch.diagonal(rho_tilde, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
        rho_bag = rho_tilde / (traces + self.eps)

        # Step 6: Measurement via Observable Operator
        O_sym = self.get_symmetric_observables()
        logits = torch.einsum('cij,bij->bc', O_sym, rho_bag)

        # Step 7: von Neumann Entropy S(ρ) = -Σ λ_i ln(λ_i)
        eigenvalues = torch.linalg.eigvalsh(rho_bag)
        eigenvalues = torch.clamp(eigenvalues, min=self.eps)
        entropy = -torch.sum(eigenvalues * torch.log(eigenvalues), dim=-1)

        return logits, entropy, rho_bag, alpha


def von_neumann_entropy_loss(entropy: torch.Tensor, clean_mask: torch.Tensor) -> torch.Tensor:
    """
    Penalizes high entropy on known-clean training bags to force the model 
    toward pure, coherent morphological states for clean samples.
    """
    if clean_mask.sum() == 0:
        return torch.tensor(0.0, device=entropy.device)
    return entropy[clean_mask].mean()


# =============================================================================
# DATASET & DATALOADER UTILITIES
# =============================================================================

class BloodSmearBagDataset(Dataset):
    """Stores patient-level bags of medical image features."""
    def __init__(self, bags_z, bag_labels):
        self.bags_z = [torch.tensor(bag, dtype=torch.float32) for bag in bags_z]
        self.bag_labels = torch.tensor(bag_labels, dtype=torch.long)

    def __len__(self):
        return len(self.bag_labels)

    def __getitem__(self, idx):
        return self.bags_z[idx], self.bag_labels[idx]


def qmil_collate_fn(batch):
    """
    Custom collate function for BloodSmearBagDataset.
    Dynamically pads bags and generates the boolean attention mask.
    """
    bags, labels = zip(*batch)
    lengths = torch.tensor([bag.size(0) for bag in bags])
    
    padded_bags = pad_sequence(bags, batch_first=True, padding_value=0.0)
    
    max_len = padded_bags.size(1)
    batch_size = len(bags)
    mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
    
    labels = torch.stack(labels)
    
    return padded_bags, labels, mask


def create_pytorch_bags(df, patch_features):
    """Groups patch-level feature matrices into patient-level bags."""
    bags_z = []
    y_bag_labels = []
    
    df_reset = df.reset_index(drop=True)
    unique_patients = df_reset['patient_id'].unique()
    
    for pid in unique_patients:
        patient_mask = (df_reset['patient_id'] == pid).values
        patient_features = patch_features[patient_mask]
        patient_label = df_reset[patient_mask]['label'].iloc[0]
        
        bags_z.append(patient_features)
        y_bag_labels.append(patient_label)
        
    return bags_z, y_bag_labels


# =============================================================================
# FEATURE EXTRACTION & EVALUATION UTILITIES
# =============================================================================

def save_feature_extractors(config, trained_models_list, layer_index=-2):
    """Creates and saves feature extractors from a list of trained models."""
    save_dir = config['MODEL_PATH'] if isinstance(config, dict) else config.MODEL_PATH
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[Saving Feature Extractors] Directory: {save_dir}")

    for fold in range(config.DATA.N_SPLIT):
        full_model = trained_models_list[fold]
        
        if isinstance(layer_index, int):
            target_layer = full_model.layers[layer_index]
        else:
            target_layer = full_model.get_layer(layer_index)
            
        print(f"  Fold {fold}: Extracting features from layer '{target_layer.name}'")

        feature_extractor = tf.keras.Model(
            inputs=full_model.input,
            outputs=target_layer.output,
            name=f"FeatureExtractor_Fold{fold}"
        )

        filename = f"fold_{fold}_feature_extractor.h5"
        fe_path = os.path.join(save_dir, filename)
        feature_extractor.save(fe_path)
        print(f"    Saved: {filename}")

    print("[Complete] All feature extractors saved.")


def extract_features(model_name, model, data_gen, fold, save_dir, prefix, 
                     suffix='train', layer_identifier=-2, verbose=False):
    """Extracts intermediate backbone features for downstream Q-MIL processing."""
    if verbose: 
        print(f"[Feature Extraction] Processing Fold {fold}...")
    
    if isinstance(layer_identifier, int):
        target_output = model.layers[layer_identifier].output
    elif isinstance(layer_identifier, str):
        target_output = model.get_layer(layer_identifier).output
    else:
        raise ValueError("layer_identifier must be an integer index or a layer name string.")
        
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=target_output) 
    feature_extractor.trainable = False
    
    features = feature_extractor.predict(
        data_gen, 
        verbose=1 if verbose else 0,
        workers=4 if platform.system() == 'Linux' else 1,
        use_multiprocessing=True if platform.system() == 'Linux' else False
    )
    
    if hasattr(data_gen, 'classes'):
        labels = np.array(data_gen.classes)
    elif hasattr(data_gen, 'labels'):
        labels = np.array(data_gen.labels)
    else:
        labels_list = [y for _, y in data_gen]
        labels = np.concatenate(labels_list, axis=0)
        if len(labels.shape) > 1: 
            labels = np.argmax(labels, axis=1)

    os.makedirs(save_dir, exist_ok=True)
    features_save_path = os.path.join(save_dir, f"{prefix}_features_{suffix}.npy" if suffix in ['train', 'val', 'test'] else f"features_{suffix}.npy")
        
    np.save(features_save_path, features)
    if verbose: 
        print(f"  -> Saved Features: {features_save_path} | Shape: {features.shape}")
    
    return feature_extractor, features, labels


def build_val_dataloader(fold, config, val_gen, model_per_fold=None, model_name="ConvNeXtLarge", 
                         target_pids=None, sample_patch=None, prefix=None):
    """Constructs a PyTorch DataLoader for the validation cohort."""
    print(f"\n[Val DataLoader] Initializing for Fold {fold}")
    bag_df = val_gen.df.copy()
    
    if target_pids is not None:
        print(f" -> [TARGET MODE] Filtering specific patient IDs: {target_pids}")
        bag_df = bag_df[bag_df['patient_id'].isin(target_pids)].reset_index(drop=True)
        
        if sample_patch is not None:
            bag_df_shuffled = bag_df.sample(frac=1, random_state=42)
            bag_df = bag_df_shuffled.groupby('patient_id').head(sample_patch).reset_index(drop=True)
        
        if bag_df.empty:
            print(f" -> [SKIP] No target patients reside in Validation Fold {fold}.")
            return None 
            
        if model_per_fold is None:
            raise ValueError("model_per_fold is strictly required for targeted live extraction.")
            
        target_gen = PatchDatasetPreserve(
            config=config, df=bag_df, batch_size=config.TRAIN.BATCH_SIZE,
            preprocessing_function=val_gen.preprocessing_function, shuffle=False
        )
        
        _, X_val_z, _ = extract_features(
            model_name=model_name, model=model_per_fold, data_gen=target_gen,
            fold=fold, save_dir=config.MODEL_PATH, layer_identifier=config.TRAIN.Feature_Layer_Index,
            prefix=prefix, suffix=f'val_targeted_{fold}', verbose=True
        )
    else:
        val_gen.shuffle = False
        if hasattr(val_gen, 'on_epoch_end'):
            val_gen.on_epoch_end()
            
        if config.DATA.Extract_Feature:
            if model_per_fold is None:
                raise ValueError("model_per_fold is required when Extract_Feature=True.")
            _, X_val_z, _ = extract_features(
                model_name=model_name, model=model_per_fold, data_gen=val_gen,
                fold=fold, save_dir=config.MODEL_PATH, layer_identifier=config.TRAIN.Feature_Layer_Index, 
                prefix=prefix, suffix='val'
            )
        else:
            val_feat_path = os.path.join(config.MODEL_PATH, f"{prefix}_features_val.npy" if prefix else f"fold_{fold}_{model_name}_features_val.npy")
            if not os.path.exists(val_feat_path):
                raise FileNotFoundError(f"Missing feature artifact: {val_feat_path}")
            X_val_z = np.load(val_feat_path)

    val_bags_z, val_y_bag_labels = create_pytorch_bags(df=bag_df, patch_features=X_val_z)
    val_dataset = BloodSmearBagDataset(val_bags_z, val_y_bag_labels)
    val_loader = DataLoader(
        val_dataset, batch_size=config.TRAIN.MIL_BATCH_SIZE, 
        shuffle=False, collate_fn=qmil_collate_fn, pin_memory=True
    )
    
    print(f" -> Compilation Complete: {len(val_dataset)} patients | {len(val_loader)} batches.")
    return val_loader


def build_test_dataloader(fold, config, test_gen, model_per_fold=None, model_name="ConvNeXtLarge", 
                          target_pids=None, sample_patch=None, prefix=None):
    """Constructs a PyTorch DataLoader for the unseen test cohort."""
    print(f"\n[Test DataLoader] Initializing for Fold {fold}")
    bag_df = test_gen.df.copy()
    
    if target_pids is not None:
        bag_df = bag_df[bag_df['patient_id'].isin(target_pids)].reset_index(drop=True)
        
        if sample_patch is not None:
            bag_df_shuffled = bag_df.sample(frac=1, random_state=42)
            bag_df = bag_df_shuffled.groupby('patient_id').head(sample_patch).reset_index(drop=True)

        if bag_df.empty:
            return None 
            
        target_gen = PatchDatasetPreserve(
            config=config, df=bag_df, batch_size=config.TRAIN.BATCH_SIZE,
            preprocessing_function=test_gen.preprocessing_function, shuffle=False
        )
        
        _, X_test_z, _ = extract_features(
            model_name=model_name, model=model_per_fold, data_gen=target_gen,
            fold=fold, save_dir=config.MODEL_PATH, layer_identifier=config.TRAIN.Feature_Layer_Index,
            prefix=prefix, suffix=f'test_targeted_{fold}', verbose=True
        )
    else:
        test_gen.shuffle = False
        if hasattr(test_gen, 'on_epoch_end'):
            test_gen.on_epoch_end()
            
        if config.DATA.Extract_Feature:
            _, X_test_z, _ = extract_features(
                model_name=model_name, model=model_per_fold, data_gen=test_gen,
                fold=fold, save_dir=config.MODEL_PATH, layer_identifier=config.TRAIN.Feature_Layer_Index, 
                prefix=prefix, suffix='test'
            )
        else:
            test_feat_path = os.path.join(config.MODEL_PATH, f"{prefix}_features_test.npy" if prefix else f"fold_{fold}_{model_name}_features_test.npy")
            if not os.path.exists(test_feat_path):
                raise FileNotFoundError(f"Missing feature artifact: {test_feat_path}")
            X_test_z = np.load(test_feat_path)

    test_bags_z, test_y_bag_labels = create_pytorch_bags(df=bag_df, patch_features=X_test_z)
    test_dataset = BloodSmearBagDataset(test_bags_z, test_y_bag_labels)
    test_loader = DataLoader(
        test_dataset, batch_size=config.TRAIN.MIL_BATCH_SIZE, 
        shuffle=False, collate_fn=qmil_collate_fn, pin_memory=True
    )
    
    print(f" -> Compilation Complete: {len(test_dataset)} patients | {len(test_loader)} batches.")
    return test_loader


# =============================================================================
# DIAGNOSTICS & PLOTTING
# =============================================================================

def sanity_check_dataset(dataset, df, num_samples=1):
    """Verifies that the PyTorch Bag Dataset exactly matches the original DataFrame."""
    unique_patients = df.reset_index(drop=True)['patient_id'].unique()
    print("\n" + "-"*40 + f"\n DATASET SANITY CHECK ({num_samples} Samples)\n" + "-"*40)
    
    for _ in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        patient_id = unique_patients[idx]
        bag_tensor, label_tensor = dataset[idx]
        
        df_patient = df[df['patient_id'] == patient_id]
        expected_patches = len(df_patient)
        expected_label = int(df_patient['label'].iloc[0])
        
        print(f"Patient ID:        {patient_id} (Index: {idx})")
        print(f"Bag Tensor Shape:  {bag_tensor.shape} -> Expected Patches: {expected_patches}")
        print(f"Bag Label:         {label_tensor.item()} -> Expected: {expected_label}")
        print(f"Contains NaNs?:    {torch.isnan(bag_tensor).any().item()}")
        
        assert bag_tensor.shape[0] == expected_patches, f"Patch count mismatch for {patient_id}!"
        assert label_tensor.item() == expected_label, f"Label mismatch for {patient_id}!"
        assert not torch.isnan(bag_tensor).any(), f"NaNs detected in features for {patient_id}!"
        print("Status:            PASSED\n")


def sanity_check_dataloader(loader, dataset, num_samples: int = 2):
    """Verifies shapes, mask integrity, and label alignment in a DataLoader batch."""
    print("\n" + "="*40 + "\n DATALOADER SANITY CHECK")
    bags, labels, mask, *rest = next(iter(loader))

    B, N_max, D = bags.shape
    print(f"\n[Batch-Level]\n  bags shape     : {list(bags.shape)}")
    print(f"  labels shape   : {list(labels.shape)}\n  mask shape     : {list(mask.shape)}")
    print(f"  Contains NaN?  : {torch.isnan(bags).any().item()}")

    real_counts = mask.sum(dim=1).tolist()
    print(f"\n[Mask Integrity]")
    for i, count in enumerate(real_counts):
        print(f"  Patient [{i}]: {count} real patches | {N_max - count} padded")

    for i in range(B):
        pad_rows = bags[i][~mask[i]]
        if pad_rows.numel() > 0:
            assert pad_rows.abs().sum().item() == 0.0, f"Patient [{i}]: padded rows are non-zero!"

    print("\n STATUS: PASSED\n")


def plot_qmil_training_history(history_data, save_dir=None, prefix="QMIL"):
    """Plots the training and validation curves from the Q-MIL training history."""
    df = pd.read_csv(history_data) if isinstance(history_data, str) else history_data.copy()
    epochs = df['Epoch']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    
    # Plot 1: Loss Curve
    ax1.plot(epochs, df['Train_Loss'], label='Train Loss', color='blue', marker='o', markersize=4)
    ax1.plot(epochs, df['Val_Loss'], label='Validation Loss', color='red', marker='s', markersize=4)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel(f"Epoch\nBest Train Loss: {df['Train_Loss'].min():.4f} | Best Val Loss: {df['Val_Loss'].min():.4f}", fontsize=12)
    ax1.set_ylabel('Loss (BCE + Entropy Reg)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: Metrics Curve
    ax2.plot(epochs, df['Val_Acc'], label='Val Accuracy', color='green', marker='^', markersize=4)
    ax2.plot(epochs, df['Val_AUC'], label='Val AUC', color='purple', marker='d', markersize=4)
    ax2.set_title('Validation Performance Metrics', fontsize=14, fontweight='bold')
    ax2.set_xlabel(f"Epoch\nBest Val Acc: {df['Val_Acc'].max():.4f} | Best Val AUC: {df['Val_AUC'].max():.4f}", fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_ylim([0.0, 1.05])
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{prefix}_training_curves.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"\n[Plot] Training curves saved to: {save_path}")
        
    plt.show()
    plt.close(fig)


def create_pilot_df(df, patches_per_patient: int = 100, patients_per_class: int = 2, seed: int = 42):
    """Creates a small pilot DataFrame with balanced classes and fixed patch counts."""
    rng = np.random.default_rng(seed)
    chunks = []

    for cls in df['label'].unique():
        cls_patients = df[df['label'] == cls]['patient_id'].unique()
        chosen_patients = rng.choice(cls_patients, size=patients_per_class, replace=False)

        for pid in chosen_patients:
            patient_df = df[(df['patient_id'] == pid) & (df['label'] == cls)]
            sampled = patient_df.sample(n=patches_per_patient, random_state=seed)
            chunks.append(sampled)

    return pd.concat(chunks, ignore_index=True)