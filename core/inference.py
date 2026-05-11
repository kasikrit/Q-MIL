import os
import torch
import pandas as pd
import numpy as np
from models.qmil_head_improved_2 import QMILHead

def run_ensemble_inference(config, test_loader, H_LIMIT, weights_dir, output_dir, device):
    """
    Evaluates the full 5-fold ensemble on the held-out test set and applies clinical triage.
    """
    print("\n" + "="*60)
    print(" PHASE 3: INDEPENDENT ENSEMBLE TEST INFERENCE")
    print("="*60)
    
    # Load all 5 models into memory
    models = []
    for fold in range(5):
        q_mil = QMILHead(d_in=config.d_in, d_model=config.d_model, num_classes=2).to(device)
        model_path = os.path.join(weights_dir, f"QMIL_Fold_{fold}.pth")
        q_mil.load_state_dict(torch.load(model_path, map_location=device))
        q_mil.eval()
        models.append(q_mil)

    results = []
    
    with torch.no_grad():
        for batch_z, batch_y, batch_mask, patient_ids in test_loader:
            batch_z, batch_mask = batch_z.to(device), batch_mask.to(device)
            
            ensemble_probs = []
            ensemble_rhos = []
            
            # Collect outputs from all models
            for model in models:
                logits, _, rho_bag, _ = model(z_raw=batch_z, mask=batch_mask)
                ensemble_probs.append(torch.softmax(logits, dim=-1))
                ensemble_rhos.append(rho_bag)
                
            # Soft-voting Probability aggregation
            mean_probs = torch.stack(ensemble_probs).mean(dim=0)
            
            # Consensus Density Matrix aggregation
            mean_rho = torch.stack(ensemble_rhos).mean(dim=0)
            
            for b in range(mean_rho.shape[0]):
                rho_sym = (mean_rho[b] + mean_rho[b].T) / 2
                eigenvalues = torch.linalg.eigvalsh(rho_sym)
                eigenvalues = torch.clamp(eigenvalues, min=1e-9, max=1.0)
                entropy = (-eigenvalues * torch.log(eigenvalues)).sum().item()
                
                prob_thl = float(mean_probs[b, 1].item())
                pred_label = 1 if prob_thl >= 0.5 else 0
                true_label = int(batch_y[b].item())
                
                # Clinical Triage Gate
                routing = "Zone A (Autonomous)" if entropy <= H_LIMIT else "Zone B (Triage/Review)"
                
                results.append({
                    'patient_id': patient_ids[b],
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'prob_THL': prob_thl,
                    'entropy': entropy,
                    'routing': routing
                })

    df_results = pd.DataFrame(results)
    csv_file = os.path.join(output_dir, "Test_Ensemble_Predictions.csv")
    df_results.to_csv(csv_file, index=False)
    
    # Calculate Yield
    zone_a_count = len(df_results[df_results['routing'] == "Zone A (Autonomous)"])
    print(f"Inference Complete. Autonomous Yield: {zone_a_count/len(df_results)*100:.2f}%")
    