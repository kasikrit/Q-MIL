import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def generate_qmil_XAI(z_raw, mask, model, patient_id, output_dir, device):
    """
    Generates the 3-panel XAI Visualization Suite:
    1. Morphological PCA
    2. Quantum Density Matrix
    3. Antagonistic Feature Activation
    """
    model.eval()
    with torch.no_grad():
        z_raw = z_raw.to(device)
        mask = mask.to(device)
        
        # Extract internal states
        Z_proj = model.projector(z_raw)
        logits, Z_ent, rho_bag, _ = model(z_raw=z_raw, mask=mask)
        
        # Convert to numpy for plotting
        Z_np = Z_proj[0][mask[0]].cpu().numpy()
        rho_np = rho_bag[0].cpu().numpy()
        
        # Retrieve Observables
        O_0 = model.observables[0].cpu()
        O_1 = model.observables[1].cpu()
        O_0_sym = ((O_0 + O_0.T) / 2).numpy()
        O_1_sym = ((O_1 + O_1.T) / 2).numpy()
        O_net = O_1_sym - O_0_sym
        
        # Interaction Matrix
        interaction_matrix = rho_np * O_net

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=300)
    
    # --------------------------------------------------
    # Panel 1: Morphological Latent Space (PCA)
    # --------------------------------------------------
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(Z_np)
    magnitudes = np.linalg.norm(Z_pca, axis=1)
    
    scatter = axes[0].scatter(Z_pca[:, 0], Z_pca[:, 1], c=magnitudes, cmap='viridis', alpha=0.7)
    axes[0].set_title("1. Morphological Latent Space (2D)")
    fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04, label="L2 Magnitude")

    # --------------------------------------------------
    # Panel 2: Quantum Latent State (Density Matrix)
    # --------------------------------------------------
    im1 = axes[1].imshow(rho_np, cmap='magma', interpolation='nearest')
    axes[1].set_title(r"2. Quantum Latent State ($\rho$)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Amplitude")

    # --------------------------------------------------
    # Panel 3: Antagonistic Feature Activation
    # --------------------------------------------------
    vmax = np.max(np.abs(interaction_matrix))
    im2 = axes[2].imshow(interaction_matrix, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    axes[2].set_title(r"3. Measurement Interaction ($\rho \odot O_{net}$)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Diagnostic Feature Competition (Blue=IDA, Red=THL)")

    plt.suptitle(f"Q-MIL XAI Report: Patient {patient_id}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"XAI_{patient_id}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved XAI plot to {save_path}")
