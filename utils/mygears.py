"""
Q-MIL Utility Gears
===================
Provides shared utility functions for the Q-MIL framework, including 
experimental reproducibility enforcement, statistical evaluation, 
calibration metrics, and plotting tools for the clinical triage gate.
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    brier_score_loss, 
    roc_curve
)
from typing import Dict, Any, Tuple


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def seed_everything(seed: int = 1337) -> None:
    """
    Enforces strict deterministic execution across all computational backends 
    to guarantee reproducibility of the Monte Carlo cross-validation splits.
    
    Args:
        seed (int): The universal random seed (default: 1337).
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    print(f"[System] Deterministic execution enforced. Seed globally set to {seed}.")


# =============================================================================
# CLINICAL TRIAGE LOGIC
# =============================================================================

def apply_clinical_triage(p_patient_mean: float, t_lower: float, t_upper: float, t_opt: float) -> Dict[str, Any]:
    """
    Applies the tri-state (Hybrid) decision logic to categorize patients into 
    Zone A (Autonomous) or Zone B (Expert Review).
    
    Args:
        p_patient_mean (float): Soft-voted ensemble probability for THL.
        t_lower (float): Lower threshold boundary for IDA confidence.
        t_upper (float): Upper threshold boundary for THL confidence.
        t_opt (float): Optimal binary threshold for mean-based fallback.
        
    Returns:
        dict: Mapping of triage status, predicted label, and diagnostic string.
    """
    # Zone A: Confident IDA
    if p_patient_mean < t_lower:
        return {
            'status': "Zone A (Autonomous)",
            'pred': 0,
            'diagnosis': "IDA"
        }
        
    # Zone A: Confident THL
    elif p_patient_mean >= t_upper:
        return {
            'status': "Zone A (Autonomous)",
            'pred': 1,
            'diagnosis': "THL"
        }
        
    # Zone B: Epistemic Uncertainty (Triage needed)
    else:
        # Fallback to binary threshold for statistical comparison purposes
        pred = 1 if p_patient_mean >= t_opt else 0
        diagnosis = "THL" if pred == 1 else "IDA"
        
        return {
            'status': "Zone B (Triage Review)",
            'pred': pred,
            'diagnosis': diagnosis
        }


# =============================================================================
# METRICS & EVALUATION
# =============================================================================

def evaluate_classification_performance(y_true: np.ndarray, y_pred: np.ndarray, 
                                        class_labels: list = ['IDA', 'THL']) -> Dict[str, float]:
    """
    Computes detailed clinical classification metrics including Sensitivity, 
    Specificity, PPV, and NPV from binary arrays.
    
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Binary predicted labels.
        class_labels (list): Names of the diagnostic classes.
        
    Returns:
        dict: Computed metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Handle edge cases where only one class is predicted
    if cm.shape == (1, 1):
        TP = cm[0, 0] if y_true[0] == 1 else 0
        TN = cm[0, 0] if y_true[0] == 0 else 0
        FP = FN = 0
    else:
        FP = cm.sum(axis=0) - np.diag(cm)  
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
    
    # Add epsilon to prevent division by zero
    eps = 1e-10
    TPR = TP / (TP + FN + eps) # Sensitivity
    TNR = TN / (TN + FP + eps) # Specificity
    PPV = TP / (TP + FP + eps) # Precision
    NPV = TN / (TN + FN + eps) # Negative Predictive Value
    
    metrics = {
        'Sensitivity': TPR[1] if len(TPR) > 1 else TPR[0],
        'Specificity': TNR[1] if len(TNR) > 1 else TNR[0],
        'Precision': PPV[1] if len(PPV) > 1 else PPV[0],
        'NPV': NPV[1] if len(NPV) > 1 else NPV[0],
        'Accuracy': (TP.sum() + eps) / (cm.sum() + eps)
    }
    
    return metrics


def compute_calibration_scores(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
    """
    Computes the Brier Score and Expected Calibration Error (ECE) to validate 
    the trustworthiness of the probability outputs in Zone A.
    
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_prob (np.ndarray): Predicted probabilities for the positive class (THL).
        n_bins (int): Number of bins for ECE calculation.
        
    Returns:
        Tuple[float, float]: Brier Score and ECE.
    """
    brier = brier_score_loss(y_true, y_prob)
    
    # Calculate Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return brier, ece


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_triage_funnel(df: pd.DataFrame, save_dir: str) -> None:
    """
    Generates a funnel plot illustrating the distribution of patients across 
    the Zone A (Autonomous) and Zone B (Triage/Review) strata.
    """
    if 'routing' not in df.columns:
        print("[Warning] Cannot plot funnel: 'routing' column missing from DataFrame.")
        return
        
    zone_counts = df['routing'].value_counts()
    
    plt.figure(figsize=(8, 6), dpi=300)
    bars = plt.bar(zone_counts.index, zone_counts.values, color=['#2ca02c', '#d62728'])
    
    plt.title("Double-Verification Triage Yield", fontsize=14, fontweight='bold')
    plt.ylabel("Number of Patients", fontsize=12)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), 
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
                 
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'triage_yield_funnel.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Triage yield funnel saved to: {save_path}")


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, title: str, save_path: str) -> None:
    """
    Plots the Receiver Operating Characteristic (ROC) curve and calculates the AUC.
    """
    try:
        auc_score = roc_auc_score(y_true, y_prob)
    except ValueError:
        print("[Warning] Skipping ROC plot: Only one class present in y_true.")
        return
        
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    