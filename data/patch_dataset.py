"""
Standard Patch Generator with Data Augmentation
===============================================
Keras Sequence generator for feeding segmented RBC patches into the 
backbone architecture during Stage 1 pre-training.

Features built-in support for CutMix (V1 and V2) to robustly regularize 
the latent space against morphological weak-label noise.
"""

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Tuple, Callable, Optional, Union


class PatchDataset(tf.keras.utils.Sequence):
    """
    Generates batches of images and labels for model training and feature extraction,
    with integrated CutMix augmentation policies.
    
    Args:
        config (CfgNode): Global configuration object (e.g., YACS config).
        df (pd.DataFrame): DataFrame containing 'image_path' and 'label' columns.
        batch_size (int): Number of samples per batch.
        preprocessing_function (Callable, optional): Backbone-specific preprocessing function.
        shuffle (bool): Whether to shuffle the data at the end of each epoch.
        use_cutmix (bool): Flag to enable CutMix augmentation.
        cutmix_alpha (float): Beta distribution parameter for CutMix area selection.
        cutmix_version (str): CutMix strategy ('v1' for standard, 'v2' for forced cross-class).
        verbose (bool): If True, prints additional logging during data loading.
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
                 verbose: bool = False):
        
        self.config = config
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.preprocessing_function = preprocessing_function
        self.indices = np.arange(len(self.df))
        
        # --- Augmentation Initialization ---
        self.use_cutmix = use_cutmix
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_version = cutmix_version
        
        if self.use_cutmix:
            if cutmix_version not in ['v1', 'v2']:
                raise ValueError("cutmix_version must be 'v1' or 'v2'.")
            if self.verbose:
                print(f"CutMix enabled (Version: {self.cutmix_version}) with alpha={self.cutmix_alpha}")

        # Initialize the first epoch
        self.on_epoch_end()

    def __len__(self) -> int:
        """Denotes the total number of batches per epoch."""
        return int(np.ceil(len(self.df) / self.batch_size))

    @property
    def classes(self) -> np.ndarray:
        """Returns the array of labels for compatibility with certain sklearn metrics."""
        return self.df['label'].to_numpy()

    def on_epoch_end(self) -> None:
        """Updates and shuffles indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generates and returns one batch of data."""
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.df))
        
        batch_indices = self.indices[start_idx:end_idx]
        batch_df = self.df.iloc[batch_indices]
        
        # 1. Load basic images and labels
        images, labels = self._load_basic_image(batch_df)
        
        # 2. Apply CutMix Augmentation (if enabled)
        if self.use_cutmix:
            images, labels = self._apply_cutmix(images, labels)
            
        # 3. Apply Backbone-Specific Preprocessing
        images = self._apply_model_preprocessing(images)
        
        return images, labels

    def _load_basic_image(self, batch_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Loads raw images from disk and performs basic resizing."""
        img_size = self.config.DATA.IMG_SIZE
        
        # Pre-allocate arrays
        images = np.empty((len(batch_df), img_size, img_size, 3), dtype=np.float32)
        labels = np.empty((len(batch_df), self.config.MODEL.NUM_CLASSES), dtype=np.float32)

        for i, (_, row) in enumerate(batch_df.iterrows()):
            img_path = str(row['image_path'])
            img = cv2.imread(img_path)
            
            if img is None:
                raise IOError(f"Failed to load image at {img_path}. Verify path integrity.")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))

            images[i,] = img
            labels[i,] = tf.keras.utils.to_categorical(
                row['label'], 
                num_classes=self.config.MODEL.NUM_CLASSES
            )

        return images, labels

    def _apply_model_preprocessing(self, images: np.ndarray) -> np.ndarray:
        """Applies the specific preprocessing function (e.g., ConvNeXt scaling)."""
        if self.preprocessing_function:
            # Handle potential TensorFlow/NumPy type mismatches from CutMix
            if isinstance(images, tf.Tensor):
                images = images.numpy()
                
            processed_images = np.empty_like(images)
            for i in range(images.shape[0]):
                processed_images[i] = self.preprocessing_function(images[i])
            return processed_images
            
        return images

    # =========================================================================
    # CutMix Augmentation Subroutines
    # =========================================================================

    def _apply_cutmix(self, images: np.ndarray, labels: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
        """Routes to the configured CutMix policy."""
        images_tf = tf.convert_to_tensor(images, dtype=tf.float32)
        labels_tf = tf.convert_to_tensor(labels, dtype=tf.float32)
        
        if self.cutmix_version == 'v1':
            return self._apply_cutmix_v1(images_tf, labels_tf)
        elif self.cutmix_version == 'v2':
            return self._apply_cutmix_v2_force_diff_class(images_tf, labels_tf)

    def _apply_cutmix_v1(self, images: tf.Tensor, labels_onehot: tf.Tensor, probability: float = 1.0) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Standard CutMix: Mixes patches randomly across the current batch without class enforcement.
        """
        batch_size = tf.shape(images)[0]
        img_h, img_w = tf.shape(images)[1], tf.shape(images)[2]

        def _cutmix_single(j):
            apply_cutmix = tf.random.uniform([], 0.0, 1.0) <= probability
            
            def _do_cutmix():
                k = tf.random.uniform([], 0, batch_size, dtype=tf.int32)
                img_j, img_k = images[j], images[k]
                label_j, label_k = labels_onehot[j], labels_onehot[k]

                lam = tf.compat.v1.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample()
                lam = tf.clip_by_value(lam, 0.2, 0.8)

                cut_w = tf.cast(tf.cast(img_w, tf.float32) * tf.sqrt(1.0 - lam), tf.int32)
                cut_h = tf.cast(tf.cast(img_h, tf.float32) * tf.sqrt(1.0 - lam), tf.int32)
                
                cx = tf.random.uniform([], 0, img_w, dtype=tf.int32)
                cy = tf.random.uniform([], 0, img_h, dtype=tf.int32)
                
                x1 = tf.clip_by_value(cx - cut_w // 2, 0, img_w)
                x2 = tf.clip_by_value(cx + cut_w // 2, 0, img_w)
                y1 = tf.clip_by_value(cy - cut_h // 2, 0, img_h)
                y2 = tf.clip_by_value(cy + cut_h // 2, 0, img_h)

                mixed_img = tf.identity(img_j)
                patch = img_k[y1:y2, x1:x2, :]

                yy, xx = tf.meshgrid(tf.range(y1, y2), tf.range(x1, x2), indexing="ij")
                coords = tf.stack([yy, xx], axis=-1)
                coords_flat = tf.reshape(coords, [-1, 2])
                patch_flat = tf.reshape(patch, [-1, tf.shape(img_j)[-1]])

                mixed_img = tf.tensor_scatter_nd_update(mixed_img, coords_flat, patch_flat)

                target_area = tf.cast((x2 - x1) * (y2 - y1), tf.float32)
                total_area = tf.cast(img_w * img_h, tf.float32)
                actual_lam = 1.0 - (target_area / total_area)
                mixed_label = actual_lam * label_j + (1.0 - actual_lam) * label_k

                return mixed_img, mixed_label

            def _skip_cutmix():
                return images[j], labels_onehot[j]

            return tf.cond(apply_cutmix, _do_cutmix, _skip_cutmix)

        elems = tf.range(batch_size)
        mixed_images, mixed_labels = tf.map_fn(
            _cutmix_single, elems, 
            dtype=(tf.float32, tf.float32)
        )
        return mixed_images, mixed_labels

    def _apply_cutmix_v2_force_diff_class(self, images: tf.Tensor, labels_onehot: tf.Tensor, probability: float = 1.0) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        CutMix V2: Forces the mixed patch to originate from an opposing diagnostic class 
        whenever possible, maximizing the regularizing effect of the augmentation.
        """
        batch_size = tf.shape(images)[0]
        img_h, img_w = tf.shape(images)[1], tf.shape(images)[2]
        labels_class = tf.argmax(labels_onehot, axis=1)

        def _cutmix_single(j):
            apply_cutmix = tf.random.uniform([], 0.0, 1.0) <= probability
            
            def _do_cutmix():
                class_j = labels_class[j]
                
                # Identify candidates of the opposing class
                diff_idx = tf.where(tf.not_equal(labels_class, class_j))[:, 0]
                no_diff = tf.equal(tf.size(diff_idx), 0)

                # Fallback to self if no opposing class is found in the current batch
                k = tf.cond(
                    no_diff,
                    lambda: j,
                    lambda: diff_idx[tf.random.uniform([], 0, tf.shape(diff_idx)[0], dtype=tf.int32)]
                )

                img_j, img_k = images[j], images[k]
                label_j, label_k = labels_onehot[j], labels_onehot[k]

                lam = tf.compat.v1.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample()
                lam = tf.clip_by_value(lam, 0.2, 0.8)

                cut_w = tf.cast(tf.cast(img_w, tf.float32) * tf.sqrt(1.0 - lam), tf.int32)
                cut_h = tf.cast(tf.cast(img_h, tf.float32) * tf.sqrt(1.0 - lam), tf.int32)
                
                cx = tf.random.uniform([], 0, img_w, dtype=tf.int32)
                cy = tf.random.uniform([], 0, img_h, dtype=tf.int32)
                
                x1 = tf.clip_by_value(cx - cut_w // 2, 0, img_w)
                x2 = tf.clip_by_value(cx + cut_w // 2, 0, img_w)
                y1 = tf.clip_by_value(cy - cut_h // 2, 0, img_h)
                y2 = tf.clip_by_value(cy + cut_h // 2, 0, img_h)

                mixed_img = tf.identity(img_j)
                patch = img_k[y1:y2, x1:x2, :]

                yy, xx = tf.meshgrid(tf.range(y1, y2), tf.range(x1, x2), indexing="ij")
                coords = tf.stack([yy, xx], axis=-1)
                coords_flat = tf.reshape(coords, [-1, 2])
                patch_flat = tf.reshape(patch, [-1, tf.shape(img_j)[-1]])

                mixed_img = tf.tensor_scatter_nd_update(mixed_img, coords_flat, patch_flat)

                target_area = tf.cast((x2 - x1) * (y2 - y1), tf.float32)
                total_area = tf.cast(img_w * img_h, tf.float32)
                actual_lam = 1.0 - (target_area / total_area)
                mixed_label = actual_lam * label_j + (1.0 - actual_lam) * label_k

                return mixed_img, mixed_label

            def _skip_cutmix():
                return images[j], labels_onehot[j]

            return tf.cond(apply_cutmix, _do_cutmix, _skip_cutmix)

        elems = tf.range(batch_size)
        mixed_images, mixed_labels = tf.map_fn(
            _cutmix_single, elems, 
            dtype=(tf.float32, tf.float32)
        )
        return mixed_images, mixed_labels

    # =========================================================================
    # Diagnostics & Visualization
    # =========================================================================

    def plot_random_batch(self, num_images: int = 4, title: str = "Random Batch Sample") -> None:
        """
        Retrieves a random batch from the sequence and plots the images and labels.
        Useful for verifying CutMix behavior and preprocessing integrity.
        """
        random_batch_index = np.random.randint(0, len(self))
        images, labels = self.__getitem__(random_batch_index)
        
        num_images = min(num_images, len(images))
        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
        if num_images == 1:
            axes = [axes]
            
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i in range(num_images):
            # De-normalize if necessary for display
            img_display = images[i]
            if img_display.min() < 0 or img_display.max() <= 1.0:
                img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-5)
            else:
                img_display = img_display.astype(np.uint8)
                
            axes[i].imshow(img_display)
            axes[i].set_title(f"Label: {np.round(labels[i], 2)}")
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()

    def summary(self) -> None:
        """Prints a statistical summary of the dataset configuration."""
        print("="*40)
        print(" PATCH DATASET SUMMARY")
        print("="*40)
        print(f" Total Samples    : {len(self.df)}")
        print(f" Batch Size       : {self.batch_size}")
        print(f" Total Batches    : {len(self)}")
        print(f" Preprocessing    : {'Enabled' if self.preprocessing_function else 'None'}")
        print(f" Shuffle Enabled  : {self.shuffle}")
        print(f" CutMix Enabled   : {self.use_cutmix} (Version: {self.cutmix_version})")
        print("="*40)
        