"""
Backbone Feature Extractor Builder
==================================
Factory class for initializing pre-trained Convolutional Neural Networks (CNNs) 
used as the classical feature extraction backbone prior to Q-MIL quantum state embedding.
"""

import tensorflow as tf
from tensorflow.keras import models, layers, applications
from typing import Tuple, Callable, Optional

class ModelBuilder:
    """
    Constructs and configures CNN architectures for hematopathology patch extraction.
    
    Args:
        model_name (str): Name of the architecture (e.g., 'ConvNeXtLarge', 'EfficientNetB7').
        input_shape (Tuple[int, int, int]): Shape of the input image patches.
        num_classes (int): Number of target classes for pre-training.
        freeze_base (bool): If True, freezes the ImageNet weights of the backbone.
    """
    def __init__(self, 
                 model_name: str, 
                 input_shape: Tuple[int, int, int] = (256, 256, 3), 
                 num_classes: int = 2,
                 freeze_base: bool = False):
        self.model_name = model_name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.freeze_base = freeze_base

    def create_model(self) -> models.Model:
        """Instantiates the specified backbone and attaches the classification head."""
        base_model = self._get_base_model()
        
        if self.freeze_base:
            base_model.trainable = False

        # Attach custom top layers for Stage 1 pre-training
        x = base_model.output
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='classifier')(x)

        return models.Model(inputs=base_model.input, outputs=outputs, name=self.model_name)

    def _get_base_model(self) -> models.Model:
        """Factory method to retrieve the requested Keras application."""
        architectures = {
            'ConvNeXtLarge': applications.ConvNeXtLarge,
            'EfficientNetB7': applications.EfficientNetB7,
            'ResNet50': applications.ResNet50
        }
        
        if self.model_name not in architectures:
            raise ValueError(f"Architecture '{self.model_name}' is not supported.")
            
        return architectures[self.model_name](
            weights='imagenet', 
            include_top=False, 
            input_shape=self.input_shape
        )

    def get_preprocessing_function(self) -> Optional[Callable]:
        """Returns the specific preprocessing function required by the backbone."""
        if 'ConvNeXt' in self.model_name:
            return applications.convnext.preprocess_input
        elif 'EfficientNet' in self.model_name:
            return applications.efficientnet.preprocess_input
        else:
            return applications.resnet.preprocess_input
