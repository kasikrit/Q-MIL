import torch
import torch.nn as nn
import torch.nn.functional as F

# Baseline 1: Hybrid Soft-Voting CNN (Deterministic)
import torch
import torch.nn as nn

class SoftVotingMIL(nn.Module):
    def __init__(self, d_in=1536, num_classes=2):
        super(SoftVotingMIL, self).__init__()
        self.classifier = nn.Linear(d_in, num_classes)
        
    def forward(self, z_raw, mask):
        # z_raw shape: [B, N, 1536], mask shape: [B, N]
        
        # 1. Patch-level logits: [B, N, 2]
        patch_logits = self.classifier(z_raw)
        
        # 2. Convert to probabilities: [B, N, 2]
        patch_probs = torch.softmax(patch_logits, dim=-1)
        
        # 3. Mask out invalid padding cells
        mask_expanded = mask.unsqueeze(-1).expand_as(patch_probs)
        valid_patch_probs = patch_probs * mask_expanded
        
        # 4. Bag-level probability is the mean of valid patch probabilities
        # CRITICAL FIX: Removed .unsqueeze(-1) so shape remains [B, 1]
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1e-9) 
        
        # [B, 2] / [B, 1] -> safely broadcasts to [B, 2]
        bag_probs = valid_patch_probs.sum(dim=1) / valid_counts
        
        # Return log(probs) to mimic logits for CrossEntropyLoss compatibility
        bag_logits = torch.log(bag_probs + 1e-9)
        return bag_logits, None, None, None # Padding returns to match QMIL signature
    
# Baseline 2: Attention-Based MIL (AB-MIL)
class Adapted_GatedAttention(nn.Module):
    def __init__(self, d_in=1536, d_hidden=128, num_classes=2):
        super(Adapted_GatedAttention, self).__init__()
        self.L = d_hidden
        self.ATTENTION_BRANCHES = 1

        # Original feature extractors (CNNs) are bypassed since z_raw is already 1536D.

        self.attention_V = nn.Sequential(
            nn.Linear(d_in, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(d_in, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w 

        # Adapted from Sigmoid to a 2-class linear output for CrossEntropyLoss compatibility
        self.classifier = nn.Sequential(
            nn.Linear(d_in * self.ATTENTION_BRANCHES, num_classes)
        )

    def forward(self, z_raw, mask):
        # z_raw shape: [B, N_cells, 1536], mask shape: [B, N_cells]
        
        A_V = self.attention_V(z_raw)  # [B, N, L]
        A_U = self.attention_U(z_raw)  # [B, N, L]
        
        # Element-wise multiplication (V * U) from the original architecture
        attn_scores = self.attention_w(A_V * A_U).squeeze(-1) # [B, N]
        
        # --- CRITICAL FIX: Apply batch mask BEFORE Softmax ---
        # Ensures padding instances receive exactly 0.0 attention
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        A = F.softmax(attn_scores, dim=1)  # [B, N]
        
        # --- Bag Representation (Weighted Sum) ---
        # Replaces original torch.mm(A, H) to support batched multiplication
        Z = torch.bmm(A.unsqueeze(1), z_raw).squeeze(1) # [B, 1536]
        
        bag_logits = self.classifier(Z) # [B, 2]
        
        # Return signature matches QMILHead so the train loop doesn't break
        return bag_logits, None, None, None


# Baseline 3: CLAM-SB (Clustering-constrained Attention MIL)
class Adapted_CLAM_SB(nn.Module):
    def __init__(self, d_in=1536, d_hidden1=512, d_hidden2=256, num_classes=2, dropout=True):
        super(Adapted_CLAM_SB, self).__init__()
        self.num_classes = num_classes

        # --- 1. Original Gated Attention Network (from model_clam.py) ---
        self.attention_a = nn.Sequential(
            nn.Linear(d_in, d_hidden1),
            nn.ReLU(),
            nn.Dropout(0.25) if dropout else nn.Identity(),
            nn.Linear(d_hidden1, d_hidden2),
            nn.Tanh()
        )
        self.attention_b = nn.Sequential(
            nn.Linear(d_in, d_hidden1),
            nn.ReLU(),
            nn.Dropout(0.25) if dropout else nn.Identity(),
            nn.Linear(d_hidden1, d_hidden2),
            nn.Sigmoid()
        )
        self.attention_c = nn.Linear(d_hidden2, 1)

        # --- 2. Original Instance-Level Classifier ---
        self.instance_classifier = nn.Linear(d_in, num_classes)

        # --- 3. Original Bag-Level Classifier ---
        self.classifiers = nn.Linear(d_in, num_classes)

    def forward(self, z_raw, mask):
        # z_raw shape: [B, N_cells, 1536], mask shape: [B, N_cells]
        
        # --- Gated Attention Calculation ---
        a = self.attention_a(z_raw) # [B, N, 256]
        b = self.attention_b(z_raw) # [B, N, 256]
        A = a.mul(b)                # [B, N, 256]
        attn_scores = self.attention_c(A).squeeze(-1) # [B, N]
        
        # --- Crucial Fix: Apply Mask BEFORE Softmax ---
        # Ensures padding instances receive exactly 0.0 attention
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        A_norm = F.softmax(attn_scores, dim=1) # [B, N]
        
        # --- Bag Representation (Weighted Sum) ---
        M = torch.bmm(A_norm.unsqueeze(1), z_raw).squeeze(1) # [B, 1536]
        
        # --- Classifications ---
        bag_logits = self.classifiers(M) # [B, 2]
        inst_logits = self.instance_classifier(z_raw) # [B, N, 2]
        
        # Return signature matches QMILHead so the train loop doesn't break
        return bag_logits, A_norm, inst_logits, None


def compute_clam_instance_loss(A_norm, inst_logits, mask, bag_labels, k_sample=8):
    """
    Replicates the exact top-k / bottom-k pseudo-labeling from CLAM.
    """
    B = A_norm.size(0)
    total_inst_loss = 0.0
    valid_batches = 0
    criterion = nn.CrossEntropyLoss()
    
    for b in range(B):
        # Only evaluate on valid, unpadded cells
        valid_A = A_norm[b][mask[b]]
        valid_logits = inst_logits[b][mask[b]]
        bag_label = bag_labels[b].item()
        
        # Skip if the patient has too few cells for the k_sample
        if valid_A.size(0) < k_sample * 2:
            continue
            
        # Identify highest and lowest attended cells
        _, indices = torch.sort(valid_A, descending=True)
        top_k_idx = indices[:k_sample]
        bottom_k_idx = indices[-k_sample:]
        
        inst_preds = torch.cat([valid_logits[top_k_idx], valid_logits[bottom_k_idx]], dim=0)
        
        # Pseudo-labeling logic from original CLAM
        if bag_label == 1: # THL (Positive Bag)
            # Top-K are THL(1), Bottom-K are IDA(0)
            top_k_labels = torch.full((k_sample,), 1, dtype=torch.long, device=A_norm.device)
            bottom_k_labels = torch.full((k_sample,), 0, dtype=torch.long, device=A_norm.device)
        else: # IDA (Negative Bag)
            # If the bag is negative, all cells must be negative (0)
            top_k_labels = torch.full((k_sample,), 0, dtype=torch.long, device=A_norm.device)
            bottom_k_labels = torch.full((k_sample,), 0, dtype=torch.long, device=A_norm.device)
            
        inst_labels = torch.cat([top_k_labels, bottom_k_labels], dim=0)
        
        total_inst_loss += criterion(inst_preds, inst_labels)
        valid_batches += 1
        
    return (total_inst_loss / valid_batches) if valid_batches > 0 else torch.tensor(0.0, device=A_norm.device)





