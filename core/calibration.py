import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
# Note: Assume build_val_dataloader is abstracted in a data_utils module
# from utils.data_utils import build_val_dataloader 
from models.qmil_head_improved_2 import QMILHead

def calibrate_thresholds(config, val_gen_list, weights_dir, output_dir, device):
    """
    Executes Out-of-Fold (OOF) evaluation to calculate the epistemic entropy boundary.
    """
    print("\n" + "="*60)
    print(" PHASE 1B: ENTROPY THRESHOLD CALIBRATION (H_LIMIT)")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)
    all_folds = list(range(5))
    S_TP = []
    calibration_results = []

    for fold in all_folds:
        print(f"\nEvaluating Unseen Validation Patients for Fold {fold}")
        
        # Initialize DataLoader and Model
        val_loader = val_gen_list[fold] # Simplified for template
        q_mil = QMILHead(d_in=config.d_in, d_model=config.d_model, num_classes=2).to(device)
        
        model_path = os.path.join(weights_dir, f"QMIL_Fold_{fold}.pth")
        q_mil.load_state_dict(torch.load(model_path, map_location=device))
        q_mil.eval()
        
        pat_counter = 0
        with torch.no_grad():
            for batch_z, batch_y, batch_mask, patient_ids in val_loader:
                batch_z, batch_y, batch_mask = batch_z.to(device), batch_y.to(device), batch_mask.to(device)
                
                logits, _, rho_bag, _ = q_mil(z_raw=batch_z, mask=batch_mask)
                probs = torch.softmax(logits, dim=-1)
                
                for b in range(rho_bag.shape[0]):
                    pid = patient_ids[b]
                    rho_sym = (rho_bag[b] + rho_bag[b].T) / 2
                    
                    # Calculate von Neumann Entropy
                    eigenvalues = torch.linalg.eigvalsh(rho_sym)
                    eigenvalues = torch.clamp(eigenvalues, min=1e-9, max=1.0)
                    S_p = (-eigenvalues * torch.log(eigenvalues)).sum().item()
                    
                    prob_thl = float(probs[b, 1].item())
                    pred_label = 1 if prob_thl >= 0.5 else 0
                    true_label = int(batch_y[b].item())
                    is_correct = (pred_label == true_label)
                    
                    if is_correct:
                        S_TP.append(S_p)
                        
                    calibration_results.append({
                        'fold': fold,
                        'patient_id': pid,
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'prob_THL': prob_thl,
                        'entropy': S_p,
                        'correct': is_correct
                    })

    # Calculate 95th Percentile H_LIMIT
    q_percentile = 95
    H_LIMIT = np.percentile(S_TP, q_percentile)
    
    csv_file = os.path.join(output_dir, "Validation_Entropies.csv")
    pd.DataFrame(calibration_results).to_csv(csv_file, index=False)
    
    print(f"Calibration Complete. Calculated H_LIMIT (q={q_percentile}): {H_LIMIT:.6f}")
    return H_LIMIT


