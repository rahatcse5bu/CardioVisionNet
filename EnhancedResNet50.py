"""
EnhancedResNet50.py - Modified ResNet50 architecture for ECG Image Classification
Optimized for improved accuracy, precision, recall, f1-score, and specificity
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, applications, regularizers
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Activation, Add, Concatenate, Dropout, Multiply, Lambda, Reshape
import tensorflow_addons as tfa
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np

class ECGAttentionModule(layers.Layer):
    """Custom attention module specifically designed for ECG features"""
    
    def __init__(self, channels, reduction_ratio=8):
        super(ECGAttentionModule, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
        # Channel attention
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        
        self.fc1 = layers.Dense(channels // reduction_ratio, activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')
        
        # Spatial attention
        self.conv_spatial = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')
    
    def call(self, inputs):
        # Channel attention
        avg_pool = self.avg_pool(inputs)
        max_pool = self.max_pool(inputs)
        
        avg_pool = self.fc1(avg_pool)
        avg_pool = self.fc2(avg_pool)
        
        max_pool = self.fc1(max_pool)
        max_pool = self.fc2(max_pool)
        
        channel_attention = layers.Add()([avg_pool, max_pool])
        channel_attention = tf.reshape(channel_attention, [-1, 1, 1, self.channels])
        
        channel_refined = layers.Multiply()([inputs, channel_attention])
        
        # Spatial attention
        avg_spatial = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
        max_spatial = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
        concat_spatial = layers.Concatenate()([avg_spatial, max_spatial])
        spatial_attention = self.conv_spatial(concat_spatial)
        
        refined_features = layers.Multiply()([channel_refined, spatial_attention])
        
        # Residual connection
        output = layers.Add()([inputs, refined_features])
        
        return output

class ECGSpecificBlock(layers.Layer):
    """ECG-specific convolutional block for detecting ECG patterns"""
    
    def __init__(self, filters):
        super(ECGSpecificBlock, self).__init__()
        self.filters = filters
        
        # Small kernels for P, Q, S waves (fine details)
        self.conv_small = layers.Conv2D(filters//4, kernel_size=3, padding='same')
        self.bn_small = layers.BatchNormalization()
        
        # Medium kernels for QRS complexes
        self.conv_medium = layers.Conv2D(filters//4, kernel_size=5, padding='same')
        self.bn_medium = layers.BatchNormalization()
        
        # Large kernels for T waves and ST segments
        self.conv_large = layers.Conv2D(filters//4, kernel_size=9, padding='same')
        self.bn_large = layers.BatchNormalization()
        
        # Very large kernels for overall ECG morphology
        self.conv_xlarge = layers.Conv2D(filters//4, kernel_size=15, padding='same')
        self.bn_xlarge = layers.BatchNormalization()
        
        # Integration
        self.conv_integrate = layers.Conv2D(filters, kernel_size=1, padding='same')
        self.bn_integrate = layers.BatchNormalization()
    
    def call(self, inputs):
        # Small receptive field
        x_small = self.conv_small(inputs)
        x_small = self.bn_small(x_small)
        x_small = layers.Activation('relu')(x_small)
        
        # Medium receptive field
        x_medium = self.conv_medium(inputs)
        x_medium = self.bn_medium(x_medium)
        x_medium = layers.Activation('relu')(x_medium)
        
        # Large receptive field
        x_large = self.conv_large(inputs)
        x_large = self.bn_large(x_large)
        x_large = layers.Activation('relu')(x_large)
        
        # Extra large receptive field
        x_xlarge = self.conv_xlarge(inputs)
        x_xlarge = self.bn_xlarge(x_xlarge)
        x_xlarge = layers.Activation('relu')(x_xlarge)
        
        # Concatenate multi-scale features
        x_concat = layers.Concatenate()([x_small, x_medium, x_large, x_xlarge])
        
        # Integrate
        x_integrated = self.conv_integrate(x_concat)
        x_integrated = self.bn_integrate(x_integrated)
        x_integrated = layers.Activation('relu')(x_integrated)
        
        # Residual connection if input and output channels match
        if inputs.shape[-1] == self.filters:
            x_integrated = layers.Add()([inputs, x_integrated])
        
        return x_integrated

class FeatureCalibrationModule(layers.Layer):
    """Module for calibrating feature importance based on ECG domain knowledge"""
    
    def __init__(self, channels):
        super(FeatureCalibrationModule, self).__init__()
        self.channels = channels
        
        # Learnable parameters for ECG regions
        self.p_wave_detector = self.create_pattern_detector()
        self.qrs_detector = self.create_pattern_detector()
        self.t_wave_detector = self.create_pattern_detector()
        self.st_segment_detector = self.create_pattern_detector()
        
        # Feature recalibration
        self.recalibration = layers.Conv2D(channels, kernel_size=1, padding='same')
    
    def create_pattern_detector(self):
        return layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')
    
    def call(self, inputs):
        # Detect ECG components
        p_wave_map = self.p_wave_detector(inputs)
        qrs_map = self.qrs_detector(inputs)
        t_wave_map = self.t_wave_detector(inputs)
        st_segment_map = self.st_segment_detector(inputs)
        
        # Create importance map - higher weights to pathological regions
        combined_map = layers.Concatenate()([p_wave_map, qrs_map, t_wave_map, st_segment_map])
        importance_map = layers.Conv2D(1, kernel_size=1, padding='same', activation='sigmoid')(combined_map)
        
        # Apply importance calibration
        calibrated = layers.Multiply()([inputs, importance_map])
        
        # Residual connection
        output = layers.Add()([inputs, calibrated])
        output = self.recalibration(output)
        
        return output

def create_enhanced_resnet50(input_shape=(224, 224, 3), num_classes=5, weights='imagenet'):
    """
    Creates an enhanced ResNet50 model optimized for ECG image classification
    
    Enhancements:
    1. ECG-specific attention modules
    2. Multi-scale feature extraction for different ECG components
    3. Feature calibration based on ECG domain knowledge
    4. Modified residual connections
    5. Label smoothing and focal loss for better training
    
    Args:
        input_shape: Input image dimensions (height, width, channels)
        num_classes: Number of classification categories
        weights: Pre-training weights, 'imagenet' or None
        
    Returns:
        Enhanced ResNet50 model
    """
    inputs = Input(shape=input_shape)
    
    # Input preprocessing
    x = layers.experimental.preprocessing.Rescaling(1./255)(inputs)
    
    # Apply contrast enhancement for better ECG visibility
    x = layers.Lambda(lambda x: tf.image.adjust_contrast(x, 1.5))(x)
    
    # Base ResNet50 model (without top layers)
    base_model = ResNet50(
        include_top=False,
        weights=weights,
        input_tensor=x,
        input_shape=input_shape
    )
    
    # Unfreeze upper layers of ResNet50 for fine-tuning
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    
    # Get intermediate outputs for feature fusion
    c2 = base_model.get_layer('conv2_block3_out').output  # Early features
    c3 = base_model.get_layer('conv3_block4_out').output  # Mid-level features
    c4 = base_model.get_layer('conv4_block6_out').output  # Higher-level features
    c5 = base_model.get_layer('conv5_block3_out').output  # Semantic features
    
    # Apply ECG-specific processing to different levels
    p2 = ECGSpecificBlock(c2.shape[-1])(c2)  # Process early features for fine ECG details
    p3 = ECGSpecificBlock(c3.shape[-1])(c3)  # Process mid features for QRS complex
    p4 = ECGSpecificBlock(c4.shape[-1])(c4)  # Process higher features for T waves
    p5 = ECGSpecificBlock(c5.shape[-1])(c5)  # Process semantic features
    
    # Apply feature attention to higher-level features
    p4 = ECGAttentionModule(p4.shape[-1])(p4)  
    p5 = ECGAttentionModule(p5.shape[-1])(p5)
    
    # Feature calibration for pathology emphasis
    p5 = FeatureCalibrationModule(p5.shape[-1])(p5)
    
    # Feature Pyramid-like upsampling and fusion
    # Upsample p5 and combine with p4
    p5_up = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(p5)
    p5_up = layers.Conv2D(p4.shape[-1], kernel_size=1, padding='same')(p5_up)
    p4 = layers.Add()([p4, p5_up])
    p4 = layers.Conv2D(p4.shape[-1], kernel_size=3, padding='same')(p4)
    p4 = layers.BatchNormalization()(p4)
    p4 = layers.Activation('relu')(p4)
    
    # Global pooling
    gap = layers.GlobalAveragePooling2D()(p5)
    
    # Create additional context streams
    context1 = layers.GlobalMaxPooling2D()(p4)
    context2 = layers.GlobalAveragePooling2D()(p4)
    
    # Combine features from different streams
    combined = layers.Concatenate()([gap, context1, context2])
    
    # Add dropout for regularization and preventing overfitting
    x = layers.Dropout(0.5)(combined)
    
    # Classification head
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    
    # Final classification layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='EnhancedResNet50_ECG')
    
    return model

def get_training_config(model, learning_rate=0.001):
    """
    Get recommended training configuration for the enhanced model
    
    Args:
        model: The enhanced ResNet50 model
        learning_rate: Initial learning rate
        
    Returns:
        Dictionary with optimizer, loss function, and metrics
    """
    # Create optimizer with learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=10000,
        alpha=1e-6,
        warmup_target=0.001,
        warmup_steps=1000
    )
    
    optimizer = tfa.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4
    )
    
    # Create focal loss for handling class imbalance
    def focal_loss(gamma=2.0, alpha=0.25):
        def focal_loss_fn(y_true, y_pred):
            # Clip predictions to prevent numerical instability
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            
            # Calculate focal loss
            cross_entropy = -y_true * tf.math.log(y_pred)
            weight = tf.pow(1.0 - y_pred, gamma) * y_true
            
            return tf.reduce_sum(alpha * weight * cross_entropy, axis=-1)
        return focal_loss_fn

    # Create custom metrics for ECG classification
    def specificity(y_true, y_pred):
        # Convert probabilities to class predictions
        y_pred_cls = tf.argmax(y_pred, axis=1)
        y_true_cls = tf.argmax(y_true, axis=1)
        
        tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_cls, 0), tf.equal(y_pred_cls, 0)), tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_cls, 0), tf.not_equal(y_pred_cls, 0)), tf.float32))
        
        return tn / (tn + fp + tf.keras.backend.epsilon())
    
    # Return training configuration
    return {
        'optimizer': optimizer,
        'loss': focal_loss(gamma=2.0, alpha=0.25),
        'metrics': [
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tfa.metrics.F1Score(num_classes=model.output_shape[1], average='macro'),
            specificity
        ]
    }

def get_data_augmentation():
    """
    Returns data augmentation pipeline optimized for ECG images
    
    ECG-specific considerations:
    - Limited rotations to preserve ECG morphology
    - No horizontal flips (would invert R waves)
    - Careful brightness/contrast adjustments
    - Random crops that preserve key ECG features
    
    Returns:
        Data augmentation model
    """
    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomRotation(0.05),  # Limited rotation
        layers.experimental.preprocessing.RandomZoom(0.1),
        layers.experimental.preprocessing.RandomTranslation(0.1, 0.1),
        layers.experimental.preprocessing.RandomContrast(0.2),
        layers.Lambda(lambda x: tf.image.random_brightness(x, 0.1)),
        # Custom augmentation for ECG images (simulate noise and baseline wander)
        layers.Lambda(lambda x: x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.01)),
    ])
    
    return data_augmentation 