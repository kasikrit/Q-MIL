"""
Quantum-Inspired Multiple Instance Learning (Q-MIL) Head
=========================================================
Improvements over baseline — focused on Step 2 (Quantum State Projection):

1. [FIX]  Added explicit d_in -> d_model projection layer in __init__.
          Without this, z_raw from ConvNeXt-Large (1536-dim) would silently
          mismatch the d_model=256 assumption used in W_q / W_k.

2. [FIX]  LayerNorm before L2-normalization to stabilize gradients when
          ||z_raw|| ≈ 0 (F.normalize gradient explodes at the boundary).

3. [IMPROVE] Learnable purity gate alpha_i per patch.
          L2 normalization alone treats every patch as a perfect pure state,
          discarding amplitude (magnitude) — a meaningful quality signal.
          A sigmoid gate learns to downweight noisy/artifact patches,
          producing a weighted mixed state that is physically more correct.

4. [IMPROVE] Stored purity weights returned from forward() for interpretability
          and potential use as an attention visualization in clinical UI.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import matplotlib.pyplot as plt
import os, platform
import tensorflow as tf
from PatchDatasetPreserve import PatchDatasetPreserve

class QMILHead(nn.Module):
    """
    Quantum-Inspired Multiple Instance Learning (Q-MIL) Head.

    Args:
        d_in        : Feature dimension from the backbone (e.g., 1536 for ConvNeXt-Large).
                      Previously missing — callers had to pre-project externally.
        d_model     : Internal latent dimension for the quantum state space (default: 256).
        num_classes : Number of diagnostic classes, e.g., 2 for {IDA, THL} (default: 2).
        d_k         : Key/Query projection dimension for the Hamiltonian attention (default: 64).
        eps         : Numerical stability floor (default: 1e-9).
    """

    def __init__(self, d_in: int = 1536, d_model: int = 256,
                 num_classes: int = 2, d_k: int = 64, eps: float = 1e-9):
        super(QMILHead, self).__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.d_k = d_k
        self.eps = eps

        # ------------------------------------------------------------------
        # Step 2 — FIX 1: Explicit backbone-to-latent projection.
        # ConvNeXt-Large emits 1536-dim vectors; d_model is typically 256.
        # This projection MUST live here so that W_q / W_k always operate on
        # d_model-shaped tensors regardless of the backbone used.
        # ------------------------------------------------------------------
        self.patch_proj = nn.Sequential(
            nn.Linear(d_in, d_model, bias=False),   # dimension reduction
            nn.LayerNorm(d_model),                   # FIX 2: stabilise grads before L2 norm
            nn.GELU(),                               # non-linear mixing
        )

        # ------------------------------------------------------------------
        # Step 2 — IMPROVEMENT: Learnable purity gate.
        # Maps each projected patch embedding to a scalar alpha_i in (0, 1).
        # alpha_i ≈ 1  →  clean, high-confidence RBC morphology (pure state).
        # alpha_i ≈ 0  →  noisy / artefact patch (state is downweighted).
        # This preserves the magnitude signal that L2-norm alone discards.
        # ------------------------------------------------------------------
        self.purity_gate = nn.Sequential(
            nn.Linear(d_model, 1, bias=True),
            nn.Sigmoid(),
        )

        # Step 3: Entanglement Coupling (Hamiltonian / Self-Attention)
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)

        # Step 5: Quantum Observables — one (d_model × d_model) matrix per class
        self.raw_observables = nn.Parameter(
            torch.Tensor(num_classes, d_model, d_model)
        )
        nn.init.xavier_uniform_(self.raw_observables)

    # ----------------------------------------------------------------------
    def get_symmetric_observables(self) -> torch.Tensor:
        """Enforce Hermitian (symmetric) observables: O_sym = 0.5*(O + O^T)."""
        return 0.5 * (self.raw_observables + self.raw_observables.transpose(1, 2))

    # ----------------------------------------------------------------------
    def forward(self, z_raw: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            z_raw : (Batch, N, d_in)  — raw patch features from backbone.
            mask  : (Batch, N) bool   — True = valid patch, False = padding.

        Returns:
            logits  : (Batch, num_classes)          — diagnostic logits.
            entropy : (Batch,)                      — von Neumann entropy per patient.
            rho     : (Batch, d_model, d_model)     — bag density matrices.
            alpha   : (Batch, N)                    — per-patch purity weights
                                                      (interpretability / triage use).
        """
        B, N, _ = z_raw.shape

        # ------------------------------------------------------------------
        # Step 2 (Improved): Quantum State Projection
        # ------------------------------------------------------------------
        # 2a. Project from backbone dimension to quantum latent space.
        z_proj = self.patch_proj(z_raw)            # (B, N, d_model)

        # -----------------------------------------------------------------
        # THE FIX: Bag-Level Centering MUST happen AFTER GELU 
        # -----------------------------------------------------------------
        if mask is not None:
            valid_counts = mask.sum(dim=1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
            z_sum  = (z_proj * mask.unsqueeze(-1)).sum(dim=1, keepdim=True)
            z_mean = z_sum / (valid_counts + 1e-9) # (B, 1, d_model)
            
            # Center the valid patches
            z_proj = torch.where(mask.unsqueeze(-1), 
                                 z_proj - z_mean, 
                                 torch.zeros_like(z_proj))
        else:
            # Fallback if no mask is provided
            z_proj = z_proj - z_proj.mean(dim=1, keepdim=True)
        # -----------------------------------------------------------------

        # 2b. Compute per-patch purity weights BEFORE discarding amplitude.
        alpha = self.purity_gate(z_proj).squeeze(-1)   # (B, N)

        # 2c. L2-normalise to pure quantum state vectors: |ψ_i⟩ = z / ‖z‖₂
        z = F.normalize(z_proj, p=2, dim=-1, eps=self.eps)  # (B, N, d_model)

        # 2d. Modulate each pure state by its learned purity scalar.
        #     This is equivalent to the weighted ket: α_i |ψ_i⟩
        #     Patches with alpha≈0 contribute negligibly to the density matrix.
        z = z * alpha.unsqueeze(-1)                    # (B, N, d_model)

        # ------------------------------------------------------------------
        # Step 3: Entanglement Coupling Generation (Hamiltonian)
        # ------------------------------------------------------------------
        Q = self.W_q(z)   # (B, N, d_k)
        K = self.W_k(z)   # (B, N, d_k)

        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.d_k ** 0.5)  # (B, N, N)

        if mask is not None:
            attn_mask = mask.unsqueeze(1).expand(-1, N, -1)
            attention_scores = attention_scores.masked_fill(~attn_mask, float('-inf'))

        W = F.softmax(attention_scores, dim=-1)        # Coupling matrix (B, N, N)

        if mask is not None:
            W = W * mask.unsqueeze(-1)

        # ------------------------------------------------------------------
        # Step 4: Bag-Level Density Matrix Construction
        # ------------------------------------------------------------------
        # FIX: Add a residual connection so the cell retains its own morphological 
        # variance while being modulated by the systemic attention context.
        z_entangled = z + torch.bmm(W, z)  # <--- THE CRITICAL FIX (z + Wz)
        
        # Now construct the density matrix
        rho_tilde = torch.bmm(z_entangled.transpose(1, 2), z_entangled)
        
        # 4c. Trace normalization to ensure Tr(ρ) = 1
        traces = (
            torch.diagonal(rho_tilde, dim1=-2, dim2=-1)
            .sum(dim=-1, keepdim=True)
            .unsqueeze(-1)
        )
        rho_bag = rho_tilde / (traces + self.eps)  # (B, d_model, d_model)

        # ------------------------------------------------------------------
        # Step 5: Measurement via Observable Operator
        # ŷ_c = Tr(O_c · ρ_bag)  via Frobenius inner product
        # ------------------------------------------------------------------
        O_sym  = self.get_symmetric_observables()                     # (C, d_model, d_model)
        logits = torch.einsum('cij,bij->bc', O_sym, rho_bag)         # (B, C)

        # ------------------------------------------------------------------
        # Step 6: von Neumann Entropy  S(ρ) = -Σ λ_i ln(λ_i)
        # ------------------------------------------------------------------
        eigenvalues = torch.linalg.eigvalsh(rho_bag)                 # (B, d_model)
        eigenvalues = torch.clamp(eigenvalues, min=self.eps)
        entropy     = -torch.sum(eigenvalues * torch.log(eigenvalues), dim=-1)  # (B,)

        return logits, entropy, rho_bag, alpha


# =============================================================================
# Optional: Von Neumann Entropy regularisation loss
# =============================================================================
def von_neumann_entropy_loss(entropy: torch.Tensor, clean_mask: torch.Tensor) -> torch.Tensor:
    """
    Penalise high entropy on known-clean training bags.
    Forces the model toward pure, coherent morphological states for clean samples.

    Args:
        entropy    : (Batch,) von Neumann entropies from QMILHead.
        clean_mask : (Batch,) bool — True for bags with no artefacts.

    Returns:
        Scalar regularisation loss.
    """
    if clean_mask.sum() == 0:
        return torch.tensor(0.0, device=entropy.device)
    return entropy[clean_mask].mean()


class BloodSmearBagDataset(Dataset):
    """
    Stores patient-level bags of medical image features.
    """
    def __init__(self, bags_z, bag_labels):
        """
        Args:
            bags_z: List of numpy arrays, where each array is (N_cells, D_features).
            bag_labels: List or array of patient-level diagnoses (0 for IDA, 1 for THL).
        """
        # Convert lists of NumPy arrays into lists of PyTorch Tensors
        self.bags_z = [torch.tensor(bag, dtype=torch.float32) for bag in bags_z]
        self.bag_labels = torch.tensor(bag_labels, dtype=torch.long)

    def __len__(self):
        return len(self.bag_labels)

    def __getitem__(self, idx):
        # Returns a single patient's unpadded feature tensor and their label
        return self.bags_z[idx], self.bag_labels[idx]


def save_feature_extractors(config, trained_models_list, layer_index=-2):
    """
    Creates and saves feature extractors from a list of trained models.

    Args:
        config: Configuration dictionary/object containing 'MODEL_PATH' and 'DATA.N_SPLIT'.
        trained_models_list (list): A list of your fully trained TF/Keras models (one per fold).
        layer_index (int or str): The index or name of the layer to use as feature output. 
                                  Defaults to -2 (usually the layer before the final Dense/Softmax).
                                  If using ConvNeXt/ResNet with 'pooling=avg', the output of the 
                                  base model is often the features, so you might check your model summary.
    """
    # Ensure the save directory exists
    # Handling config['MODEL_PATH'] vs config.MODEL_PATH based on your object type
    try:
        save_dir = config['MODEL_PATH']
    except TypeError:
        save_dir = config.MODEL_PATH
        
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[Saving Feature Extractors] Saving to: {save_dir}")

    for fold in range(config.DATA.N_SPLIT):
        # 1. Get the trained full model for this fold
        full_model = trained_models_list[fold]
        
        # 2. Identify the output layer
        # If layer_index is an integer (e.g., -2), get layer by index
        # If it's a string (e.g., 'global_average_pooling2d'), get by name
        if isinstance(layer_index, int):
            target_layer = full_model.layers[layer_index]
        else:
            target_layer = full_model.get_layer(layer_index)
            
        print(f"  Fold {fold}: extracting features from layer '{target_layer.name}'")

        # 3. Create the Feature Extractor Model
        feature_extractor = tf.keras.Model(
            inputs=full_model.input,
            outputs=target_layer.output,
            name=f"FeatureExtractor_Fold{fold}"
        )

        # 4. Save to Disk
        filename = f"fold_{fold}_feature_extractor.h5"
        fe_path = os.path.join(save_dir, filename)
        
        feature_extractor.save(fe_path)
        print(f"    Saved: {filename}")

    print("[Done] All feature extractors saved.")



def create_pytorch_bags(df, patch_features):
    """
    Groups patch-level feature matrices into patient-level bags for Q-MIL.
    
    Args:
        df: Pandas DataFrame for the specific fold (e.g., train_df_fold).
        patch_features: Numpy array of shape (N_patches, feature_dim) aligned exactly with df rows.
    
    Returns:
        bags_z: List of numpy arrays, each of shape (N_cells, feature_dim)
        y_bag_labels: List of integer labels
    """
    bags_z = []
    y_bag_labels = []
    
    # Ensure sequential matching between the filtered DataFrame and the features array
    df_reset = df.reset_index(drop=True)
    unique_patients = df_reset['patient_id'].unique()
    
    for pid in unique_patients:
        # Create a boolean mask for the current patient
        patient_mask = (df_reset['patient_id'] == pid).values
        
        # Slice the features matrix
        patient_features = patch_features[patient_mask]
        
        # Extract the single patient-level label
        patient_label = df_reset[patient_mask]['label'].iloc[0]
        
        bags_z.append(patient_features)
        y_bag_labels.append(patient_label)
        
    return bags_z, y_bag_labels

def sanity_check_dataset(dataset, df, num_samples=1):
    """
    Verifies that the PyTorch Bag Dataset exactly matches the original DataFrame.
    """
    # unique() preserves the exact order used in create_pytorch_bags
    unique_patients = df.reset_index(drop=True)['patient_id'].unique()
    
    print("\n" + "-"*40)
    print(f" DATASET SANITY CHECK ({num_samples} Random Patients)")
    print("-"*40)
    
    for _ in range(num_samples):
        # 1. Pick a random patient index
        idx = np.random.randint(0, len(dataset))
        patient_id = unique_patients[idx]
        
        # 2. Extract from PyTorch Dataset
        bag_tensor, label_tensor = dataset[idx]
        
        # 3. Extract Ground Truth from DataFrame
        df_patient = df[df['patient_id'] == patient_id]
        expected_num_patches = len(df_patient)
        expected_label = int(df_patient['label'].iloc[0])
        
        # 4. Print Comparison
        print(f"Patient ID:        {patient_id} (Dataset Index: {idx})")
        print(f"Bag Tensor Shape:  {bag_tensor.shape} -> Expected Patches: {expected_num_patches}")
        print(f"Bag Label:         {label_tensor.item()} -> Expected Label: {expected_label}")
        print(f"Tensor Data Type:  {bag_tensor.dtype}")
        print(f"Contains NaNs?:    {torch.isnan(bag_tensor).any().item()}")
        
        # 5. Hard Assertions (Will crash the script if a bug exists)
        assert bag_tensor.shape[0] == expected_num_patches, f"CRITICAL: Patch count mismatch for {patient_id}!"
        assert label_tensor.item() == expected_label, f"CRITICAL: Label mismatch for {patient_id}!"
        assert not torch.isnan(bag_tensor).any(), f"CRITICAL: NaN values detected in features for {patient_id}!"
        
        print("Status:            PASSED\n")

def qmil_collate_fn(batch):
    """
    Custom collate function for BloodSmearBagDataset.
    Dynamically pads bags and generates the boolean attention mask.
    """
    # batch is a list of tuples: [(bag_tensor_1, label_1), (bag_tensor_2, label_2), ...]
    bags, labels = zip(*batch)
    
    # 1. Record the true length (number of patches) of each bag before padding
    lengths = torch.tensor([bag.size(0) for bag in bags])
    
    # 2. Dynamic Padding
    # pad_sequence stacks the list of tensors and pads them with 0.0
    # to match the length of the longest sequence IN THIS BATCH.
    # Resulting shape: (Batch_Size, Max_Len_In_Batch, 1536)
    padded_bags = pad_sequence(bags, batch_first=True, padding_value=0.0)
    
    # 3. Construct the Boolean Attention Mask
    # We compare a range tensor to the lengths tensor to broadcast a mask.
    # Output Shape: (Batch_Size, Max_Len_In_Batch)
    # True = Valid patch, False = Zero-padding (to be ignored by QMILHead)
    max_len = padded_bags.size(1)
    batch_size = len(bags)
    mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
    
    # 4. Stack labels into a single tensor: Shape (Batch_Size,)
    labels = torch.stack(labels)
    
    return padded_bags, labels, mask

def plot_qmil_training_history(history_data, save_dir=None, prefix="QMIL"):
    """
    Plots the training and validation curves from the Q-MIL training history.
    """
    if isinstance(history_data, str):
        df = pd.read_csv(history_data)
    else:
        df = history_data.copy()
        
    epochs = df['Epoch']

    avg_val_acc = df['Val_Acc'].mean()
    avg_val_auc = df['Val_AUC'].mean()
    
    best_train_loss = df['Train_Loss'].min() 
    best_val_loss = df['Val_Loss'].min()     
    best_val_acc = df['Val_Acc'].max()
    best_val_auc = df['Val_AUC'].max()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    
    # Plot 1: Loss Curve
    ax1.plot(epochs, df['Train_Loss'], label='Train Loss', color='blue', marker='o', markersize=4)
    ax1.plot(epochs, df['Val_Loss'], label='Validation Loss', color='red', marker='s', markersize=4)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    
    # Append only best metrics to the X-label
    ax1_xlabel = f"Epoch\n\nBest Train Loss: {best_train_loss:.4f}  |  Best Val Loss: {best_val_loss:.4f}"
    ax1.set_xlabel(ax1_xlabel, fontsize=12)
    ax1.set_ylabel('Loss (BCE + Entropy Reg)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    if len(epochs) <= 20: ax1.set_xticks(epochs)

    # Plot 2: Metrics Curve
    ax2.plot(epochs, df['Val_Acc'], label='Val Accuracy', color='green', marker='^', markersize=4)
    ax2.plot(epochs, df['Val_AUC'], label='Val AUC', color='purple', marker='d', markersize=4)
    ax2.set_title('Validation Performance Metrics', fontsize=14, fontweight='bold')
    
    # Append only best metrics to the X-label
    ax2_xlabel = (f"Epoch\n\n"
                  f"Avg Val Acc: {avg_val_acc:.4f}  |  Avg Val AUC: {avg_val_auc:.4f}\n"
                  f"Best Val Acc: {best_val_acc:.4f}  |  Best Val AUC: {best_val_auc:.4f}")
    ax2.set_xlabel(ax2_xlabel, fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_ylim([0.0, 1.05])
    ax2.legend(loc='lower right', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    if len(epochs) <= 20: ax2.set_xticks(epochs)

    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{prefix}_training_curves.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"\n[Plot] Training curves saved to: {save_path}")
        
    plt.show()
    plt.close(fig)
    
    
# =============================================================================
# DATALOADER SANITY CHECK
# =============================================================================
def sanity_check_dataloader(loader, dataset, num_samples: int = 2):
    """
    Pulls one batch from the DataLoader and verifies shapes, mask integrity,
    label alignment, and NaN/Inf presence.

    Args:
        loader   : DataLoader (test_loader)
        dataset  : Underlying BloodSmearBagDataset (test_dataset)
        num_samples : How many random patients to inspect individually.
    """
    print("\n" + "="*40)
    print(" DATALOADER SANITY CHECK")

    # ------------------------------------------------------------------
    # 1. Single batch check
    # ------------------------------------------------------------------
    bags, labels, mask, *rest = next(iter(loader))
    is_clean = rest[0] if rest else None

    B, N_max, D = bags.shape
    print(f"\n[Batch-Level]")
    print(f"  bags shape     : {list(bags.shape)}  (B, N_max, D)")
    print(f"  labels shape   : {list(labels.shape)} — values: {labels.tolist()}")
    print(f"  mask shape     : {list(mask.shape)}")
    print(f"  dtype          : {bags.dtype}")
    print(f"  Contains NaN?  : {torch.isnan(bags).any().item()}")
    print(f"  Contains Inf?  : {torch.isinf(bags).any().item()}")
    if is_clean is not None:
        print(f"  is_clean flags : {is_clean.tolist()}")

    # Mask integrity — real patch count per patient
    real_counts = mask.sum(dim=1).tolist()
    print(f"\n[Mask Integrity]")
    for i, count in enumerate(real_counts):
        padded = N_max - count
        print(f"  Patient [{i}]: {count} real patches | "
              f"{padded} padded | label={labels[i].item()}")

    # Padding rows should be exactly zero
    for i in range(B):
        pad_rows = bags[i][~mask[i]]          # (n_padded, D)
        if pad_rows.numel() > 0:
            assert pad_rows.abs().sum().item() == 0.0, \
                f"Patient [{i}]: padded rows are non-zero!"
    print(f"  Padding values : all zeros ✓")

    # ------------------------------------------------------------------
    # 2. Random individual patient check (directly from dataset)
    # ------------------------------------------------------------------
    print(f"\n[Individual Patient Check — {num_samples} random samples]")
    indices = torch.randperm(len(dataset))[:num_samples].tolist()

    for idx in indices:
        item     = dataset[idx]
        bag      = item[0]   # (N_i, D)
        label    = item[1]
        n_patches, d_feat = bag.shape

        # Cross-reference with loader batch to verify collate alignment
        print(f"  Dataset Index  : {idx}")
        print(f"  Bag Shape      : {list(bag.shape)} -> {n_patches} patches, {d_feat}-dim")
        print(f"  Label          : {label.item()}")
        print(f"  dtype          : {bag.dtype}")
        print(f"  Contains NaN?  : {torch.isnan(bag).any().item()}")
        print(f"  Contains Inf?  : {torch.isinf(bag).any().item()}")
        print(f"  Feature range  : [{bag.min():.4f}, {bag.max():.4f}]")

        # Verify mask would be correct if this patient were batched alone
        single_bag, single_lbl, single_mask, *_ = qmil_collate_fn([dataset[idx]])
        assert single_mask.sum().item() == n_patches, \
            f"Mask mismatch: expected {n_patches}, got {single_mask.sum().item()}"
        print(f"  Mask check     : {single_mask.sum().item()} / "
              f"{single_mask.shape[1]} = ✓")

    print("\n STATUS: PASSED\n")
    
def create_pilot_df(df, patches_per_patient: int = 100, patients_per_class: int = 2, seed: int = 42):
    """
    Creates a small pilot DataFrame with balanced classes and fixed patch counts.

    Args:
        df                 : Full test_df with columns as above.
        patches_per_patient: Number of patches to sample per patient.
        patients_per_class : Number of patients to sample per class.
        seed               : Reproducibility seed.

    Returns:
        pilot_df : Balanced DataFrame ready for BloodSmearBagDataset.
    """
    rng = np.random.default_rng(seed)
    chunks = []

    for cls in df['label'].unique():
        # All patients belonging to this class
        cls_patients = df[df['label'] == cls]['patient_id'].unique()

        assert len(cls_patients) >= patients_per_class, \
            f"Class {cls} only has {len(cls_patients)} patients, " \
            f"need {patients_per_class}."

        # Sample N patients for this class
        chosen_patients = rng.choice(cls_patients, size=patients_per_class, replace=False)

        for pid in chosen_patients:
            patient_df = df[(df['patient_id'] == pid) & (df['label'] == cls)]

            assert len(patient_df) >= patches_per_patient, \
                f"Patient {pid} only has {len(patient_df)} patches, " \
                f"need {patches_per_patient}."

            # Sample N patches for this patient
            sampled = patient_df.sample(n=patches_per_patient, random_state=seed)
            chunks.append(sampled)

    pilot_df = pd.concat(chunks, ignore_index=True)
    return pilot_df

#%%
def extract_features(
        model_name,
        model,
        data_gen,
        fold,
        save_dir,
        prefix,
        suffix='train',
        layer_identifier=-2,
        verbose=False):
    """
    Step 1: Creates a feature extractor and processes the dataset to get Z.
    Input: Trained CNN model, Data Generator.
    Output: feature_extractor model, features (Z), labels (y).
    """  
    if verbose: print(f"[P1.1] Building Feature Extractor & Extracting Features for Fold {fold}...")
    
    # 1. Define Feature Extractor
    # Flexibly handles either an integer index (e.g., 303) or a string name (e.g., 'dense_2')
    if isinstance(layer_identifier, int):
        target_output = model.layers[layer_identifier].output
    elif isinstance(layer_identifier, str):
        target_output = model.get_layer(layer_identifier).output
    else:
        raise ValueError("layer_identifier must be an integer index or a layer name string.")
        
    feature_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=target_output
    ) 
    feature_extractor.trainable = False
    
    if verbose: 
        print(f">> Target Layer: {target_output.name} | Shape: {target_output.shape}")
        print(">> Extracting features...")
        
    # 2. Extract Features (Z)
    features = feature_extractor.predict(
        data_gen, 
        verbose=1 if verbose else 0,
        workers=4 if platform.system() == 'Linux' else 1,
        use_multiprocessing=True if platform.system() == 'Linux' else False
    )
    
    # 3. Get labels (Safely handle different generator types)
    if hasattr(data_gen, 'classes'):
        labels = np.array(data_gen.classes)
    elif hasattr(data_gen, 'labels'):
        labels = np.array(data_gen.labels)
    else:
        # Fallback: iterate generator strictly bounded by its length
        labels_list = []
        for i in range(len(data_gen)):
            _, y = data_gen[i]
            labels_list.append(y)
        labels = np.concatenate(labels_list, axis=0)
        if len(labels.shape) > 1: 
            labels = np.argmax(labels, axis=1)

    if verbose: print(f"  -> Extracted shape: {features.shape}, Labels shape: {labels.shape}")
    
    # 4. Save Artifacts
    os.makedirs(save_dir, exist_ok=True)
    
    if suffix == 'train' or suffix == 'val' or suffix == 'test':
        features_save_path = os.path.join(save_dir, f"{prefix}_features_{suffix}.npy")
    else:
        features_save_path = os.path.join(save_dir, f"features_{suffix}.npy")
        
    np.save(features_save_path, features)
    if verbose: print(f"Saved Features: {features_save_path}")
    
    return feature_extractor, features, labels

#%
def build_val_dataloader(fold, config, val_gen, model_per_fold=None, model_name="ConvNeXtLarge", target_pids=None, sample_patch=None, prefix=None):
    """
    Constructs a PyTorch DataLoader for the validation cohort of a specified fold.
    Integrates dynamic patient filtering, optional patch sampling, and forces live 
    feature extraction when specific patient IDs are targeted.
    """
    print(f"\n[Val DataLoader] Initializing for Fold {fold}")
    
    # 1. Patient-Level Filtering (Do this FIRST to save memory)
    bag_df = val_gen.df.copy()
    
    # =========================================================================
    # TARGET MODE: Isolate specific patients and FORCE live feature extraction
    # =========================================================================
    if target_pids is not None:
        print(f" -> [TARGET MODE] Filtering down to specific patient IDs: {target_pids}")
        bag_df = bag_df[bag_df['patient_id'].isin(target_pids)].reset_index(drop=True)
        
        # Sub-sample patches per patient
        if sample_patch is not None:
            print(f" -> [TARGET MODE] Sampling up to {sample_patch} patches per targeted patient...")
            # 1. Shuffle the entire target DataFrame randomly
            bag_df_shuffled = bag_df.sample(frac=1, random_state=42)
            
            # 2. Group by patient and take the top N patches (which are now random)
            # This completely avoids .apply() warnings and preserves all columns!
            bag_df = bag_df_shuffled.groupby('patient_id').head(sample_patch).reset_index(drop=True)
        
        if bag_df.empty:
            print(f" -> [SKIP] None of the target patients reside in Validation Fold {fold}.")
            return None 
            
        if model_per_fold is None:
            raise ValueError("model_per_fold is strictly required for targeted live feature extraction.")
            
        # Create a lightweight subset generator for ONLY the targeted patients
        print(f" -> [LIVE EXTRACTION] Compiling features for {len(bag_df)} patches...")
        from PatchDatasetPreserve import PatchDatasetPreserve
        target_gen = PatchDatasetPreserve(
            config=config,
            df=bag_df,
            batch_size=config.TRAIN.BATCH_SIZE,
            preprocessing_function=val_gen.preprocessing_function, # Inherit from base generator
            shuffle=False
        )
        
        # Force feature extraction on the subset
        _, X_val_z, _ = extract_features(
            model_name=model_name, 
            model=model_per_fold, 
            data_gen=target_gen,
            fold=fold, 
            save_dir=config.MODEL_PATH, 
            layer_identifier=config.TRAIN.Feature_Layer_Index,
            prefix=prefix, # <--- Updated to use passed prefix
            suffix=f'val_targeted_{fold}',
            verbose=True,
        )
        print(f"{X_val_z.shape=}")
        
    # =========================================================================
    # STANDARD MODE: Process the entire fold (Live Extraction vs. Disk)
    # =========================================================================
    else:
        val_gen.shuffle = False
        if hasattr(val_gen, 'on_epoch_end'):
            val_gen.on_epoch_end()
            
        if config.DATA.Extract_Feature:
            if model_per_fold is None:
                raise ValueError("model_per_fold is strictly required when Extract_Feature=True.")
            
            _, X_val_z, _ = extract_features(
                model_name=model_name, 
                model=model_per_fold, 
                data_gen=val_gen,
                fold=fold, 
                save_dir=config.MODEL_PATH, 
                layer_identifier=config.TRAIN.Feature_Layer_Index, 
                prefix=prefix, # <--- Updated to use passed prefix
                suffix='val'
            )
        else:
            # Safely resolve the filepath based on whether a prefix was provided
            val_feat_path = os.path.join(
                config.MODEL_PATH, 
                f"{prefix}_features_val.npy" if prefix else f"fold_{fold}_{model_name}_features_val.npy"
            )
            if not os.path.exists(val_feat_path):
                raise FileNotFoundError(f"Validation feature artifact missing: {val_feat_path}")
            
            X_val_z = np.load(val_feat_path)
            print(f"Loaded: {val_feat_path}")

    # 2. Q-MIL Bag Construction
    # The bag_df and X_val_z are now perfectly aligned, whether targeted or standard
    val_bags_z, val_y_bag_labels = create_pytorch_bags(df=bag_df, patch_features=X_val_z)
    
    # 3. PyTorch Dataset & Dataloader Assembly
    val_dataset = BloodSmearBagDataset(val_bags_z, val_y_bag_labels)
    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.TRAIN.MIL_BATCH_SIZE, 
        shuffle=False, 
        collate_fn=qmil_collate_fn, 
        pin_memory=True
    )
    
    print(f" -> DataLoader successfully compiled: {len(val_dataset)} patients across {len(val_loader)} batches.")
    return val_loader

#%%
def build_test_dataloader(fold, config, test_gen, model_per_fold=None, model_name="ConvNeXtLarge", target_pids=None, sample_patch=None, prefix=None):
    """
    Constructs a PyTorch DataLoader for the unseen test cohort of a specified fold.
    Integrates dynamic patient filtering, optional patch sampling, and forces live 
    feature extraction when specific patient IDs are targeted.
    """
    print(f"\n[Test DataLoader] Initializing for Fold {fold}")
    
    # 1. Patient-Level Filtering (Do this FIRST to save memory)
    bag_df = test_gen.df.copy()
    
    # =========================================================================
    # TARGET MODE: Isolate specific test patients and FORCE live feature extraction
    # =========================================================================
    if target_pids is not None:
        print(f" -> [TARGET MODE] Filtering down to specific test patient IDs: {target_pids}")
        bag_df = bag_df[bag_df['patient_id'].isin(target_pids)].reset_index(drop=True)
        
        # Sub-sample patches per patient using the optimized shuffle-and-head method
        if sample_patch is not None:
            print(f" -> [TARGET MODE] Sampling up to {sample_patch} patches per targeted test patient...")
            # Shuffle first, then take the top N to safely sample without .apply() warnings
            bag_df_shuffled = bag_df.sample(frac=1, random_state=42)
            bag_df = bag_df_shuffled.groupby('patient_id').head(sample_patch).reset_index(drop=True)

        if bag_df.empty:
            print(f" -> [SKIP] None of the target patients reside in the Test Set.")
            return None 
            
        if model_per_fold is None:
            raise ValueError("model_per_fold is strictly required for targeted live feature extraction.")
            
        # Create a lightweight subset generator for ONLY the targeted test patients
        print(f" -> [LIVE EXTRACTION] Compiling test features for {len(bag_df)} patches...")
        target_gen = PatchDatasetPreserve(
            config=config,
            df=bag_df,
            batch_size=config.TRAIN.BATCH_SIZE,
            preprocessing_function=test_gen.preprocessing_function, 
            shuffle=False
        )
        
        # Force feature extraction on the subset (using 'test_targeted' suffix)
        _, X_test_z, _ = extract_features(
            model_name=model_name, 
            model=model_per_fold, 
            data_gen=target_gen,
            fold=fold, 
            save_dir=config.MODEL_PATH, 
            layer_identifier=config.TRAIN.Feature_Layer_Index,
            prefix=prefix, # <--- Updated to use passed prefix
            suffix=f'test_targeted_{fold}', # Safely isolated from validation data
            verbose=True,
        )
        
    # =========================================================================
    # STANDARD MODE: Process the entire test set (Live Extraction vs. Disk)
    # =========================================================================
    else:
        test_gen.shuffle = False
        if hasattr(test_gen, 'on_epoch_end'):
            test_gen.on_epoch_end()
            
        if config.DATA.Extract_Feature:
            if model_per_fold is None:
                raise ValueError("model_per_fold is strictly required when Extract_Feature=True.")
            
            _, X_test_z, _ = extract_features(
                model_name=model_name, 
                model=model_per_fold, 
                data_gen=test_gen,
                fold=fold, 
                save_dir=config.MODEL_PATH, 
                layer_identifier=config.TRAIN.Feature_Layer_Index, 
                prefix=prefix, # Uses Keras prefix
                suffix='test',
                verbose=True,
            )
        else:
            # Load from disk using the exact Keras prefix         
            if not prefix:
                raise ValueError("A prefix must be provided to locate the correct .npy files.")
                
            test_feat_path = os.path.join(config.MODEL_PATH, f"{prefix}_features_test.npy")
            
            if not os.path.exists(test_feat_path):
                raise FileNotFoundError(f"Test feature artifact missing: {test_feat_path}")
            else:
                print(f"Loading features from: {test_feat_path}")
                X_test_z = np.load(test_feat_path)
                print(f"Feature shape loaded: {X_test_z.shape}")
    
        # 2. Q-MIL Bag Construction
        # bag_df should be assigned from test_gen.df
        bag_df = test_gen.df 
        test_bags_z, test_y_bag_labels = create_pytorch_bags(df=bag_df, patch_features=X_test_z)
        
        # 3. PyTorch Dataset & Dataloader Assembly
        test_dataset = BloodSmearBagDataset(test_bags_z, test_y_bag_labels)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.TRAIN.MIL_BATCH_SIZE, 
            shuffle=False, 
            collate_fn=qmil_collate_fn, 
            pin_memory=True
        )
        
        print(f" -> Test DataLoader successfully compiled: {len(test_dataset)} patients across {len(test_loader)} batches.")
        return test_loader

#%%













