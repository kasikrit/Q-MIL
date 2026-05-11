import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score
from models.qmil_head_improved_2 import QMILHead
from data.data_utils import build_train_dataloader, build_val_dataloader

def run_training_pipeline(cfg, data_dir, weights_dir, device):
    """
    Executes the Monte Carlo Cross-Validation training loop for the Q-MIL ensemble.
    Trains independent Q-MIL heads across specified folds and saves the best weights.
    """
    os.makedirs(weights_dir, exist_ok=True)
    
    # 1. Iterate through Monte Carlo Folds
    for fold in cfg.TRAIN.FOLDS:
        logging.info(f"\n{'='*50}")
        logging.info(f" STARTING TRAINING: FOLD {fold}")
        logging.info(f"{'='*50}")
        
        # Load dataframes for this specific fold
        fold_train_df = train_dfs[fold]
        fold_val_df = val_dfs[fold]

        # Build the loaders
        train_loader = build_train_dataloader(
            fold=fold, cfg=cfg, train_df=fold_train_df, model_per_fold=None
        )
        val_loader = build_val_dataloader(
            fold=fold, cfg=cfg, val_df=fold_val_df, model_per_fold=None
        )
        
        # 3. Initialize Model, Optimizer, and Loss Function
        model = QMILHead(
            d_in=cfg.MODEL.D_IN, 
            d_model=cfg.MODEL.D_MODEL, 
            num_classes=cfg.MODEL.NUM_CLASSES
        ).to(device)
        
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=cfg.TRAIN.LR_QMIL, 
            weight_decay=cfg.TRAIN.WEIGHT_DECAY
        )
        
        # CrossEntropyLoss expects logits of shape [B, C] and labels of shape [B]
        criterion = nn.CrossEntropyLoss()
        
        best_val_auc = 0.0
        
        # 4. Epoch Loop
        for epoch in range(cfg.TRAIN.EPOCHS):
            model.train()
            train_loss = 0.0
            
            # --- TRAINING PASS ---
            for batch_z, batch_y, batch_mask in train_loader:
                batch_z = batch_z.to(device)
                batch_y = batch_y.to(device)
                batch_mask = batch_mask.to(device)
                
                optimizer.zero_grad()
                
                # Forward Pass
                logits, entropy, rho_bag, alpha = model(z_raw=batch_z, mask=batch_mask)
                
                # Compute Loss
                loss = criterion(logits, batch_y)
                
                # Optional: Add von Neumann entropy regularization if desired
                # e.g., loss = loss + (lambda_reg * von_neumann_entropy_loss(entropy, clean_mask))
                
                # Backward Pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            
            # --- VALIDATION PASS ---
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                # Assuming val_loader returns patient_ids as the 4th element based on earlier code
                for batch_z, batch_y, batch_mask, _ in val_loader:
                    batch_z = batch_z.to(device)
                    batch_y = batch_y.to(device)
                    batch_mask = batch_mask.to(device)
                    
                    logits, _, _, _ = model(z_raw=batch_z, mask=batch_mask)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item()
                    
                    # Extract probabilities for the THL class (index 1)
                    probs = torch.softmax(logits, dim=-1)[:, 1]
                    all_preds.extend(probs.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
                    
            val_loss /= len(val_loader)
            val_auc = roc_auc_score(all_targets, all_preds)
            
            # Binarize predictions for accuracy calculation (threshold = 0.5)
            val_preds_bin = [1 if p >= 0.5 else 0 for p in all_preds]
            val_acc = accuracy_score(all_targets, val_preds_bin)
            
            logging.info(f"Fold {fold} | Epoch {epoch+1:02d}/{cfg.TRAIN.EPOCHS} | "
                         f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                         f"Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")
                         
            # 5. Checkpoint Saving (Save best model based on AUC)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                save_path = os.path.join(weights_dir, f"QMIL_Fold_{fold}.pth")
                torch.save(model.state_dict(), save_path)
                logging.info(f"  --> Best model updated. Saved to {save_path}")
                
        logging.info(f"=== Fold {fold} Complete. Best Validation AUC: {best_val_auc:.4f} ===")
