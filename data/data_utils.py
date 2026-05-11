"""
Data Utilities for Q-MIL
========================
Handles the bridge between TensorFlow image generators (PatchDatasetPreserve),
classical feature extraction, and PyTorch bag-level DataLoaders.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader

# Import the core PyTorch dataset utilities we defined in qmil_head_improved_2.py
from models.qmil_head_improved_2 import (
    BloodSmearBagDataset, 
    qmil_collate_fn, 
    create_pytorch_bags, 
    extract_features
)
from data.patch_dataset_preserve import PatchDatasetPreserve

def build_train_dataloader(fold, cfg, train_df, model_per_fold=None):
    """
    Constructs the PyTorch DataLoader for the training cohort of a specific fold.
    
    Args:
        fold (int): Current Monte Carlo fold index.
        cfg (CfgNode): YACS configuration node.
        train_df (pd.DataFrame): DataFrame containing the training patch paths and labels.
        model_per_fold (tf.keras.Model, optional): Loaded TF backbone for live feature extraction.
        
    Returns:
        torch.utils.data.DataLoader: Batched PyTorch dataloader yielding bags, labels, and masks.
    """
    print(f"\n[Train DataLoader] Initializing for Fold {fold}...")
    
    # 1. Initialize the TensorFlow Image Generator
    # We set shuffle=False here because feature extraction MUST perfectly align with the DataFrame rows.
    # We will apply shuffling later at the PyTorch Bag level.
    train_gen = PatchDatasetPreserve(
        df=train_df,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        img_size=cfg.DATA.IMG_SIZE,
        shuffle=False 
    )
    
    # 2. Extract or Load Latent Features
    if cfg.DATA.EXTRACT_FEATURE:
        if model_per_fold is None:
            raise ValueError("model_per_fold must be provided when EXTRACT_FEATURE is True.")
            
        print(f" -> Live feature extraction via {cfg.MODEL.BACKBONE_NAME}...")
        _, X_train_z, _ = extract_features(
            model_name=cfg.MODEL.BACKBONE_NAME, 
            model=model_per_fold, 
            data_gen=train_gen,
            fold=fold, 
            save_dir=cfg.SYSTEM.WEIGHTS_DIR, 
            layer_identifier=-2, 
            prefix=f"QMIL_Fold_{fold}",
            suffix='train'
        )
    else:
        # Load pre-extracted features from disk
        feat_path = os.path.join(cfg.SYSTEM.WEIGHTS_DIR, f"QMIL_Fold_{fold}_features_train.npy")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Feature artifact missing: {feat_path}. Set EXTRACT_FEATURE=True.")
        
        print(f" -> Loading pre-extracted features from {feat_path}")
        X_train_z = np.load(feat_path)

    # 3. Group patches into Patient-Level Bags
    train_bags_z, train_y_bag_labels = create_pytorch_bags(df=train_gen.df, patch_features=X_train_z)
    
    # 4. Assemble the PyTorch DataLoader
    train_dataset = BloodSmearBagDataset(train_bags_z, train_y_bag_labels)
    
    # CRITICAL: shuffle=True is required here so the model doesn't memorize patient order
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.TRAIN.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=qmil_collate_fn, 
        pin_memory=True
    )
    
    print(f" -> Train DataLoader compiled: {len(train_dataset)} patients across {len(train_loader)} batches.")
    return train_loader


def build_val_dataloader(fold, cfg, val_df, model_per_fold=None):
    """
    Constructs the PyTorch DataLoader for the validation cohort of a specific fold.
    """
    print(f"\n[Val DataLoader] Initializing for Fold {fold}...")
    
    # 1. Initialize the TensorFlow Image Generator
    val_gen = PatchDatasetPreserve(
        df=val_df,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        img_size=cfg.DATA.IMG_SIZE,
        shuffle=False
    )
    
    # 2. Extract or Load Latent Features
    if cfg.DATA.EXTRACT_FEATURE:
        if model_per_fold is None:
            raise ValueError("model_per_fold must be provided when EXTRACT_FEATURE is True.")
            
        print(f" -> Live feature extraction via {cfg.MODEL.BACKBONE_NAME}...")
        _, X_val_z, _ = extract_features(
            model_name=cfg.MODEL.BACKBONE_NAME, 
            model=model_per_fold, 
            data_gen=val_gen,
            fold=fold, 
            save_dir=cfg.SYSTEM.WEIGHTS_DIR, 
            layer_identifier=-2, 
            prefix=f"QMIL_Fold_{fold}",
            suffix='val'
        )
    else:
        feat_path = os.path.join(cfg.SYSTEM.WEIGHTS_DIR, f"QMIL_Fold_{fold}_features_val.npy")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Feature artifact missing: {feat_path}. Set EXTRACT_FEATURE=True.")
            
        print(f" -> Loading pre-extracted features from {feat_path}")
        X_val_z = np.load(feat_path)

    # 3. Group patches into Patient-Level Bags
    val_bags_z, val_y_bag_labels = create_pytorch_bags(df=val_gen.df, patch_features=X_val_z)
    
    # 4. Assemble the PyTorch DataLoader
    val_dataset = BloodSmearBagDataset(val_bags_z, val_y_bag_labels)
    
    # CRITICAL: shuffle=False for validation to maintain order for tracking metrics
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.TRAIN.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=qmil_collate_fn, 
        pin_memory=True
    )
    
    print(f" -> Val DataLoader compiled: {len(val_dataset)} patients across {len(val_loader)} batches.")
    return val_loader
    