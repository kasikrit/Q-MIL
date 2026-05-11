"""
Preserving Patch Generator
==========================
Extension of the standard PatchDataset designed to preserve the inherent 
morphology of RBCs. Supports loading standard RGB images or compositing 
pre-segmented RGBA images over a standardized background to isolate cellular 
features prior to quantum latent projection.
"""

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Callable, Optional, Union

# Import the base class
from data.patch_dataset import PatchDataset


class PatchDatasetPreserve(PatchDataset):
    """
    Keras Sequence generator that preserves cellular morphology. 
    Can dynamically blend segmented RGBA patches onto a black background 
    to remove confounding background artifacts, or load standard RGB images.
    
    Args:
        config (CfgNode): Global configuration object defining DATASET paths.
        df (pd.DataFrame): DataFrame containing 'patch_path' and 'label' columns.
        batch_size (int): Number of samples per batch.
        preprocessing_function (Callable, optional): Backbone-specific preprocessing function.
        shuffle (bool): Whether to shuffle the data at the end of each epoch.
        use_cutmix (bool): Flag to enable CutMix augmentation.
        cutmix_alpha (float): Beta distribution parameter for CutMix area selection.
        cutmix_version (str): CutMix strategy ('v1' for standard, 'v2' for forced cross-class).
        expect_rgba (bool): If True, processes images as 4-channel PNGs, blending the 
                            alpha channel over a solid background to isolate the cell.
    """
    
    def __init__(self, 
                 config, 
                 df: pd.DataFrame, 
                 batch_size: int = 32,
                 preprocessing_function: Optional[Callable] = None,
                 shuffle: bool = True,
                 use_cutmix: bool = False,
                 cutmix_alpha: float = 1.0,
                 cutmix_version: str = 'v1',
                 expect_rgba: bool = True):
        
        # Initialize base attributes
        self.expect_rgba = expect_rgba
        
        # Call the parent class constructor
        super().__init__(
            config=config,
            df=df,
            batch_size=batch_size,
            preprocessing_function=preprocessing_function,
            shuffle=shuffle,
            use_cutmix=use_cutmix,
            cutmix_alpha=cutmix_alpha,
            cutmix_version=cutmix_version,
            verbose=False
        )

    def _data_generation(self, batch_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads images from disk using either RGBA compositing or standard RGB scaling.
        Overrides the base PatchDataset._data_generation method.
        """
        img_size = self.config.DATA.IMG_SIZE
        
        # Pre-allocate arrays
        images = np.empty((len(batch_df), img_size, img_size, 3), dtype=np.float32)
        labels = np.empty((len(batch_df), self.config.MODEL.NUM_CLASSES), dtype=np.float32)

        for i, (_, row) in enumerate(batch_df.iterrows()):
            if self.expect_rgba:
                img = self._load_and_blend_rgba(row)
            else:
                img = self._load_and_preprocess_rgb(row)
                
            images[i,] = img
            labels[i,] = tf.keras.utils.to_categorical(
                row['label'], 
                num_classes=self.config.MODEL.NUM_CLASSES
            )

        return images, labels

    def _load_and_blend_rgba(self, row: pd.Series, bg_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """
        Loads a 4-channel RGBA image and composites it over a solid background color.
        This isolates the segmented cell and sets all transparent space to the bg_color.
        """
        root = Path(self.config.DATASET) 
        img_path = str(row['patch_path'])
        full_path = root / img_path
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found on disk: {full_path}")
            
        rgba = cv2.imread(str(full_path), cv2.IMREAD_UNCHANGED)
        
        if rgba is None:
            raise IOError(f"Failed to load image. Corrupt or invalid file: {full_path}")
            
        if len(rgba.shape) < 3 or rgba.shape[2] != 4:
            raise ValueError(f"Expected RGBA image, but received shape {rgba.shape} at {full_path}")
    
        # Separate channels (OpenCV loads as BGRA)
        rgb = rgba[..., :3].astype(np.float32)
        alpha = rgba[..., 3].astype(np.float32) / 255.0
        alpha = np.stack([alpha] * 3, axis=-1)
    
        # Create solid background array (default black)
        bg = np.array(bg_color, dtype=np.float32)
        
        # Alpha Compositing: (Source * Alpha) + (Background * (1 - Alpha))
        blended_rgb = rgb * alpha + bg * (1.0 - alpha)
        blended_rgb = blended_rgb.astype(np.uint8)
        
        # Ensure target size
        if blended_rgb.shape[0] != self.config.DATA.IMG_SIZE:
            blended_rgb = cv2.resize(
                blended_rgb, 
                (self.config.DATA.IMG_SIZE, self.config.DATA.IMG_SIZE), 
                interpolation=cv2.INTER_AREA
            )
            
        return blended_rgb

    def _load_and_preprocess_rgb(self, row: pd.Series) -> np.ndarray:
        """
        Loads a standard 3-channel BGR image and converts it to RGB.
        """
        root = Path(self.config.DATASET) 
        img_path = str(row['patch_path'])
        full_path = root / img_path
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found on disk: {full_path}")
            
        img = cv2.imread(str(full_path), cv2.IMREAD_COLOR)

        if img is None:
            raise IOError(f"Failed to load image at {full_path}")
       
        # Convert BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Ensure target size
        if img.shape[0] != self.config.DATA.IMG_SIZE:
            img = cv2.resize(
                img, 
                (self.config.DATA.IMG_SIZE, self.config.DATA.IMG_SIZE), 
                interpolation=cv2.INTER_AREA
            )

        return img.astype(np.float32)
        