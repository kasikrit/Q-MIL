"""
Configuration Node for Q-MIL Hematopathology Pipeline
Author: Kasikrit Damkliang (kasikrit.d@psu.ac.th), Prince of Songkla University
Description: Defines the default hyperparameter configurations, path routing, 
             and model architecture dimensions using YACS.
"""

import os
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# 1. System & Path Settings
# -----------------------------------------------------------------------------
_C.SYSTEM = CN()
_C.SYSTEM.SEED = 1337                  # Enforces strict reproducibility
_C.SYSTEM.VERBOSE = 1

# Relative paths for repository structure
_C.SYSTEM.BASE_DIR = "./"
_C.SYSTEM.DATA_DIR = os.path.join(_C.SYSTEM.BASE_DIR, "data")
_C.SYSTEM.WEIGHTS_DIR = os.path.join(_C.SYSTEM.BASE_DIR, "models")
_C.SYSTEM.OUTPUT_DIR = os.path.join(_C.SYSTEM.BASE_DIR, "results")

# -----------------------------------------------------------------------------
# 2. Data Pipeline Settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.IMG_SIZE = 256                 # Standard patch resolution
_C.DATA.VAL_RATIO = 0.30               # 70/30 Train/Val split per MC fold
_C.DATA.N_SPLIT = 5                    # 5-Fold Monte Carlo cross-validation
_C.DATA.EXTRACT_FEATURE = False        # Set True if pre-extracting backbone embeddings

# -----------------------------------------------------------------------------
# 3. Model Architecture (Q-MIL & Backbone)
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.BACKBONE_NAME = "ConvNeXtLarge"
_C.MODEL.D_IN = 1536                   # Feature dimension from ConvNeXt-Large
_C.MODEL.D_MODEL = 256                 # Quantum latent projection dimension
_C.MODEL.NUM_CLASSES = 2               # Binary classification (IDA vs. THL)

# -----------------------------------------------------------------------------
# 4. Training Hyperparameters
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.FOLDS = [0, 1, 2, 3, 4]       # Active folds for ensemble training
_C.TRAIN.EPOCHS = 60
_C.TRAIN.BATCH_SIZE = 32

# Learning Rates & Optimization
_C.TRAIN.LR_BACKBONE = 1e-5
_C.TRAIN.LR_QMIL = 1e-4
_C.TRAIN.WEIGHT_DECAY = 1e-4

# Bag/Patch Configurations
_C.TRAIN.MAX_BAG_SIZE = 1000           # Maximum cells per patient bag
_C.TRAIN.MIN_BAG_SIZE = 100            # Minimum cells required for valid density matrix

# -----------------------------------------------------------------------------
# 5. Execution Flags (Overridden via CLI during execution)
# -----------------------------------------------------------------------------
_C.FLAGS = CN()
_C.FLAGS.PILOT_RUN = False             # Rapid test with small subset
_C.FLAGS.RUN_TRAIN = True              # Phase 1: Train Folds
_C.FLAGS.RUN_CALIBRATION = False       # Phase 2: OOF Entropy Calibration
_C.FLAGS.RUN_INFERENCE = False         # Phase 3: Ensemble Test Inference
_C.FLAGS.RUN_XAI = False               # Phase 4: Generate Wave Function Collapse Plots


def get_cfg_defaults():
    """Returns a clone of the default config node."""
    return _C.clone()

def update_config(cfg, args):
    """
    Updates the configuration node based on command line arguments.
    Example: python main.py --batch_size 64
    """
    cfg.defrost()
    
    # Example logic mapping argparse arguments to YACS config
    if hasattr(args, 'batch_size') and args.batch_size:
        cfg.TRAIN.BATCH_SIZE = args.batch_size
    if hasattr(args, 'data_dir') and args.data_dir:
        cfg.SYSTEM.DATA_DIR = args.data_dir
        
    cfg.freeze()
    return cfg

