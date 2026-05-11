"""
Quantum-Inspired Multiple Instance Learning (Q-MIL) for Hematopathology Triage
Author: Kasikrit Damkliang (kasikrit.d@psu.ac.th), Prince of Songkla University
Description: Training, Calibration, and Inference pipeline for IDA vs. THL classification
             using von Neumann entropy-driven double-verification triage.
"""

import os
import argparse
import logging
import torch

# Custom module imports
from config import get_cfg_defaults, update_config
from core.calibration import calibrate_thresholds
from core.inference import run_ensemble_inference
from core.xai_engine import generate_qmil_XAI
from models.qmil_head_improved_2 import QMILHead
from core.training import run_training_pipeline

def parse_args():
    """Parses command-line arguments for reproducible execution."""
    parser = argparse.ArgumentParser(description="Q-MIL Hematopathology Pipeline")
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['train', 'calibrate', 'test', 'xai'], required=True,
                        help="Execution mode: train backbone, calibrate H_LIMIT, test ensemble, or generate XAI.")
    
    # Pathing
    parser.add_argument('--data_dir', type=str, default='./data/',
                        help="Root directory for blood smear patches.")
    parser.add_argument('--weights_dir', type=str, default='./models/',
                        help="Directory containing trained fold weights.")
    parser.add_argument('--output_dir', type=str, default='./results/',
                        help="Directory to save predictions, metrics, and XAI plots.")
    
    # Q-MIL Hyperparameters
    parser.add_argument('--d_in', type=int, default=1536, help="Input feature dimension (e.g., ConvNeXt).")
    parser.add_argument('--d_model', type=int, default=256, help="Quantum latent projection dimension.")
    parser.add_argument('--seed', type=int, default=1337, help="Random seed for reproducibility.")
    
    return parser.parse_args()

def set_seed(seed):
    """Enforces strict reproducibility across PyTorch and NumPy."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()

    # Load defaults and apply CLI overrides
    cfg = get_cfg_defaults()
    cfg = update_config(cfg, args)
      
    set_seed(cfg.SYSTEM.SEED)
    
    os.makedirs(cfg.SYSTEM.OUTPUT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    
    logging.info(f"Initializing Q-MIL Pipeline in '{args.mode.upper()}' mode on {device}.")

    # ---------------------------------------------------------
    # PHASE 1: Training 
    # ---------------------------------------------------------
    if args.mode == 'train':
        logging.info("Starting Monte Carlo Cross-Validation Training...")
        run_training_pipeline(
            cfg=cfg,
            data_dir=cfg.SYSTEM.DATA_DIR,
            weights_dir=cfg.SYSTEM.WEIGHTS_DIR,
            device=device
        )

    # ---------------------------------------------------------
    # PHASE 2: OOF Calibration (Extracts H_LIMIT)
    # ---------------------------------------------------------
    elif args.mode == 'calibrate':
        logging.info("Phase 2: Out-of-Fold Calibration for Entropy Thresholds.")
        
        H_LIMIT = calibrate_thresholds(
            data_dir=cfg.SYSTEM.DATA_DIR,
            weights_dir=cfg.SYSTEM.WEIGHTS_DIR,
            output_dir=cfg.SYSTEM.OUTPUT_DIR,
            device=device
        )
        logging.info(f"Calibration Complete. Calculated H_LIMIT: {H_LIMIT:.6f}")

    # ---------------------------------------------------------
    # PHASE 3: Independent Test Evaluation
    # ---------------------------------------------------------
    elif args.mode == 'test':
        logging.info("Phase 3: Strict Evaluation on Held-Out Test Set.")
        run_ensemble_inference(
            data_dir=cfg.SYSTEM.DATA_DIR,
            weights_dir=cfg.SYSTEM.WEIGHTS_DIR,
            output_dir=cfg.SYSTEM.OUTPUT_DIR,
            device=device
        )
        
    # ---------------------------------------------------------
    # PHASE 4: Explainable AI Generation
    # ---------------------------------------------------------
    elif args.mode == 'xai':
        logging.info("Phase 4: Generating Wave Function Collapse Visualizations.")
        generate_qmil_XAI(
            data_dir=cfg.SYSTEM.DATA_DIR, 
            output_dir=cfg.SYSTEM.OUTPUT_DIR, 
            device=device
        )

if __name__ == "__main__":
    main()