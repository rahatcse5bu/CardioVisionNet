# CardioVisionNet Complete Implementation for ECG Image-based CVD Prediction
# Author: Research Team
# Description: Advanced deep learning model for cardiovascular disease prediction from ECG images

# Import necessary libraries
import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.keras.layers import Layer, Conv1D, Conv2D, SeparableConv2D, BatchNormalization, Activation, Add, Input
from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D, GlobalAveragePooling2D, Concatenate, Reshape
from tensorflow.keras.regularizers import l2
import scipy.signal as signal
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import urllib.request
import glob
import zipfile
import tempfile
import shutil
import json

# Dynamically import optional dependencies
try:
    import cv2
except ImportError:
    print("OpenCV not installed. Installing it for image processing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
    import cv2

try:
    import seaborn as sns
except ImportError:
    print("Seaborn not installed. Installing it for visualization...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
        import seaborn as sns
    except:
        print("Failed to install seaborn. Visualizations may be limited.")

try:
    # Only import these if available
    from tensorflow.keras.applications import EfficientNetV2B0, ResNet50, DenseNet121
except ImportError:
    print("Some TensorFlow applications may not be available. Using alternatives if needed.")

# Check for GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPU is available: {physical_devices}")
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("Memory growth enabled on GPU devices")
    except:
        print("Memory growth couldn't be enabled")
else:
    print("No GPU found, running on CPU")
    
# Set up TensorFlow to use mixed precision for faster training
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled for faster training")
except:
    print("Mixed precision not enabled")

#----------------------- CardioVisionNet Implementation -----------------------#

class CardioVisionNet:
    """
    CardioVisionNet: Advanced deep learning architecture for ECG image-based CVD prediction
    
    A multi-modal, multi-pathway neural network architecture that integrates computer vision,
    deep learning, and physiologically-informed components for robust cardiovascular disease 
    classification from ECG images.
    """
    
    def __init__(self, 
                 input_shape=(224, 224, 3),  # Changed to image format (height, width, channels)
                 num_classes=5, 
                 learning_rate=0.001,
                 weight_decay=0.0001,
                 dropout_rate=0.3,
                 filters_base=64,
                 use_self_supervision=True,
                 use_attention=True,
                 backbone='efficientnet',
                 model_dir='model_checkpoints'):
        """
        Initialize the CardioVisionNet model for ECG image-based CVD prediction
        
        Args:
            input_shape: Shape of input ECG image (height, width, channels)
            num_classes: Number of CVD classification categories
            learning_rate: Initial learning rate for optimizer
            weight_decay: Weight decay coefficient for regularization
            dropout_rate: Dropout rate for uncertainty estimation
            filters_base: Base number of filters for convolutional layers
            use_self_supervision: Whether to use self-supervised pre-training
            use_attention: Whether to use attention mechanisms
            backbone: CNN backbone architecture ('efficientnet', 'resnet', 'densenet')
            model_dir: Directory to save model checkpoints
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.filters_base = filters_base
        self.use_self_supervision = use_self_supervision
        self.use_attention = use_attention
        self.backbone = backbone
        self.model_dir = model_dir
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Build the model
        self.model = self._build_model()
        
        # Create self-supervised model if needed
        if self.use_self_supervision:
            self.pretraining_model = self._build_self_supervised_model()
    
    def _build_model(self):
        """Constructs the complete CardioVisionNet architecture for ECG images"""
        inputs = Input(shape=self.input_shape, name="ecg_image_input")
        
        # 1. Image preprocessing and enhancement module
        x = self._image_preprocessing_module(inputs)
        
        # 2. Multi-pathway feature extraction
        path1 = self._cnn_backbone_pathway(x)
        path2 = self._local_pattern_pathway(x)
        path3 = self._wavelet_transform_pathway(x)
        
        # 3. Multi-modal fusion with attention
        fused_features = self._cross_attention_fusion([path1, path2, path3])
        
        # 4. Transformer encoder for spatial-temporal context
        context_features = self._transformer_encoder_block(fused_features)
        
        # 5. Physiologically-informed attention (simulated for ECG regions)
        if self.use_attention:
            attended_features = self._physiological_attention(context_features)
        else:
            attended_features = context_features
        
        # 6. Meta-learning adaptation module
        adaptive_features = self._meta_learning_adaptation(attended_features)
        
        # Classification features
        classification_features = layers.Dense(256, activation='swish')(adaptive_features)
        classification_features = layers.Dropout(self.dropout_rate)(classification_features)
        
        # 7. NEW: Explainable Decision Pathways with Uncertainty Quantification
        main_logits, uncertainty, explanation_logits, component_uncertainties, component_weights = self._decision_pathway_explainer(classification_features)
        
        # HRV prediction branch (additional clinical metric)
        hrv_features = layers.Dense(64, activation='swish')(adaptive_features)
        hrv_prediction = layers.Dense(1, name="hrv_prediction")(hrv_features)
        
        # Explainability output (Grad-CAM compatible)
        gradcam_features = layers.Dense(64, activation='swish', name='gradcam_features')(adaptive_features)
        gradcam_output = layers.Dense(1, name="gradcam_output")(gradcam_features)
        
        # Main classification output with uncertainty calibration
        calibrated_logits = layers.Lambda(
            lambda inputs: inputs[0] * (1 - inputs[1]),
            name="calibrated_logits"
        )([main_logits, uncertainty])
        
        main_output = layers.Activation('softmax', name="cvd_prediction")(calibrated_logits)
        
        # Create component-specific outputs for interpretability
        component_outputs = {}
        for component, logits in explanation_logits.items():
            component_outputs[f"{component}_prediction"] = layers.Activation('softmax', name=f"{component}_prediction")(logits)
        
        # Combine all outputs
        outputs = [
            main_output,          # CVD classification
            uncertainty,          # Prediction uncertainty
            hrv_prediction,       # Heart rate variability estimate
            gradcam_output,       # For explainability
            component_weights     # Component weights for interpretability
        ]
        
        # Add component-specific predictions to outputs
        for name, output in component_outputs.items():
            outputs.append(output)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Build loss dictionary with main outputs
        loss_dict = {
            "cvd_prediction": self._adaptive_focal_loss,
            "uncertainty": 'binary_crossentropy',
            "hrv_prediction": 'mean_squared_error',
            "gradcam_output": 'mean_squared_error',
            "component_weights": 'categorical_crossentropy'  # Optional - can be supervised if you have component labels
        }
        
        # Add losses for component predictions - can be used for regularization
        for name in component_outputs.keys():
            loss_dict[name] = 'categorical_crossentropy'
        
        # Loss weights - focus mainly on the primary prediction task
        loss_weights = {
            "cvd_prediction": 1.0,
            "uncertainty": 0.2,
            "hrv_prediction": 0.3,
            "gradcam_output": 0.0,
            "component_weights": 0.1
        }
        
        # Add small weights for component predictions
        for name in component_outputs.keys():
            loss_weights[name] = 0.05
        
        # Metrics for main prediction
        metrics_dict = {
            "cvd_prediction": [
                'accuracy', 
                self._sensitivity, 
                self._specificity, 
                self._f1_score,
                tf.keras.metrics.AUC(name='auc')
            ],
            "hrv_prediction": [tf.keras.metrics.MeanAbsoluteError()]
        }
        
        # Compile with specialized loss and metrics
        model.compile(
            optimizer=self._build_optimizer(),
            loss=loss_dict,
            loss_weights=loss_weights,
            metrics=metrics_dict
        )
        
        return model

    def _build_self_supervised_model(self):
        """Build a multi-view contrastive self-supervised pretraining model for ECG images"""
        # Create encoder backbone
        input_shape = self.input_shape
        
        def _create_encoder():
            inputs = Input(shape=input_shape)
            x = self._image_preprocessing_module(inputs)
            x = self._cnn_backbone_pathway(x)
            
            # Project to embedding space
            embedding = layers.Dense(256, activation=None)(x)
            embedding = tf.nn.l2_normalize(embedding, axis=1)
            return Model(inputs=inputs, outputs=embedding, name="ssl_encoder")
        
        encoder = _create_encoder()
        
        # Create inputs for four different views
        temporal_view = Input(shape=input_shape, name="temporal_view")
        morphological_view = Input(shape=input_shape, name="morphological_view") 
        frequency_view = Input(shape=input_shape, name="frequency_view")
        phase_view = Input(shape=input_shape, name="phase_view")
        
        # Get embeddings from each view
        temporal_emb = encoder(temporal_view)
        morphological_emb = encoder(morphological_view)
        frequency_emb = encoder(frequency_view)
        phase_emb = encoder(phase_view)
        
        # Create model with multi-view inputs and outputs
        ssl_model = Model(
            inputs=[temporal_view, morphological_view, frequency_view, phase_view],
            outputs=[temporal_emb, morphological_emb, frequency_emb, phase_emb]
        )
        
        # Custom multi-view contrastive loss
        def multi_view_contrastive_loss(y_true, y_pred):
            # Unpack embeddings from different views
            embeddings = y_pred
            batch_size = tf.shape(embeddings[0])[0]
            
            # Calculate similarity matrix for all pairs of views
            loss = 0
            temperature = 0.1
            
            # Loop through all pairs of views
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    # Calculate cosine similarity between all samples
                    sim_matrix = tf.matmul(embeddings[i], tf.transpose(embeddings[j])) / temperature
                    
                    # Positive pairs are the corresponding samples in different views
                    positive_mask = tf.eye(batch_size)
                    
                    # InfoNCE loss
                    row_loss = tf.nn.softmax_cross_entropy_with_logits(
                        labels=positive_mask, logits=sim_matrix)
                    col_loss = tf.nn.softmax_cross_entropy_with_logits(
                        labels=positive_mask, logits=tf.transpose(sim_matrix))
                    
                    # Combine losses from both directions
                    loss += (tf.reduce_mean(row_loss) + tf.reduce_mean(col_loss)) / 2
            
            return loss / (len(embeddings) * (len(embeddings) - 1) / 2)
        
        # Compile with contrastive loss
        ssl_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=multi_view_contrastive_loss
        )
        
        return ssl_model, encoder

    def _signal_preprocessing_module(self, inputs):
        """Advanced signal preprocessing with domain-specific filters"""
        # For the paper/prototype version, use simpler but still effective filters
        # Bandpass filter simulation using Conv1D
        x = layers.Conv1D(self.filters_base, 15, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('tanh')(x)  # Tanh to simulate bandpass behavior
        x = layers.Conv1D(self.input_shape[-1], 15, padding='same')(x)
        
        # Residual connection to maintain signal integrity
        x = layers.Add()([inputs, x])
        
        # Normalization with learnable parameters
        x = layers.LayerNormalization()(x)
        
        return x
    
    def _temporal_pathway(self, x):
        """Process temporal features of the ECG signal"""
        # Deep residual network optimized for temporal patterns
        x = self._temporal_residual_block(x, self.filters_base, 3)
        x = layers.MaxPooling1D(2)(x)
        
        x = self._temporal_residual_block(x, self.filters_base*2, 3)
        x = layers.MaxPooling1D(2)(x)
        
        x = self._temporal_residual_block(x, self.filters_base*4, 3)
        x = layers.MaxPooling1D(2)(x)
        
        # Temporal attention mechanism
        x = self._temporal_attention_module(x)
        
        # Global context
        temporal_pool = layers.GlobalAveragePooling1D()(x)
        
        return temporal_pool
    
    def _morphological_pathway(self, x):
        """Extract morphological features from ECG waveforms"""
        # Use advanced depthwise separable convolutions
        x = layers.SeparableConv1D(self.filters_base, 5, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x)
        x = layers.MaxPooling1D(2)(x)
        
        # Extract P-QRS-T wave characteristics
        x = self._cardiac_cycle_detector(x)
        
        # Implement morphology-specific feature extractors
        p_wave_features = self._wave_feature_extractor(x, 'p_wave')
        qrs_features = self._wave_feature_extractor(x, 'qrs')
        t_wave_features = self._wave_feature_extractor(x, 't_wave')
        
        # Concatenate wave-specific features
        morphological_features = layers.Concatenate()([p_wave_features, qrs_features, t_wave_features])
        
        return morphological_features
    
    def _frequency_pathway(self, x):
        """Process frequency domain features"""
        # Simplified FFT approach
        x_complex = layers.Lambda(lambda x: tf.signal.rfft(x))(x)
        x_mag = layers.Lambda(lambda x: tf.abs(x))(x_complex)
        x_phase = layers.Lambda(lambda x: tf.math.angle(x))(x_complex)
        
        # Log-scaled magnitude
        x_mag = layers.Lambda(lambda x: tf.math.log(x + 1e-6))(x_mag)
        
        # Concatenate magnitude and phase
        x = layers.Concatenate()([x_mag, x_phase])
        
        # Frequency-specific convolutional layers
        x = Conv1D(self.filters_base, 7, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv1D(self.filters_base*2, 5, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        return x
    
    def _phase_space_pathway(self, x):
        """Novel phase-space transformation pathway"""
        # Simplified implementation for prototype
        # Project into lower dimension to make the operations feasible
        projection = layers.Conv1D(64, 1, padding='same')(x)
        projection = layers.MaxPooling1D(4)(projection)
        
        # Create delayed versions for phase space coords (simple delay embedding)
        delay1 = layers.Cropping1D((0, 2))(projection)
        padding1 = layers.ZeroPadding1D((2, 0))(tf.zeros_like(delay1))
        delay1 = layers.Concatenate(axis=1)([padding1, delay1])
        
        delay2 = layers.Cropping1D((0, 4))(projection)
        padding2 = layers.ZeroPadding1D((4, 0))(tf.zeros_like(delay2))
        delay2 = layers.Concatenate(axis=1)([padding2, delay2])
        
        # Combine original + delayed to create phase space coordinates
        phase_space = layers.Concatenate()([projection, delay1, delay2])
        
        # Extract features from this phase space representation
        x = layers.Conv1D(self.filters_base*2, 5, padding='same')(phase_space)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        return x
    
    def _cross_attention_fusion(self, feature_paths):
        """Fuse multiple feature pathways using cross-attention"""
        # Ensure all features have same dimensions
        processed_features = []
        for i, path in enumerate(feature_paths):
            # Reshape if needed
            if len(path.shape) == 2:
                path = tf.expand_dims(path, axis=1)
                
            # Project to common dimension
            path_processed = layers.Dense(256, name=f"projection_{i}")(path)
            processed_features.append(path_processed)
        
        # Concatenate along sequence dimension
        if len(processed_features) > 1:
            concatenated = layers.Concatenate(axis=1)(processed_features)
        else:
            concatenated = processed_features[0]
        
        # Multi-head self-attention for cross-pathway interactions
        attention_output = layers.MultiHeadAttention(
            num_heads=8, key_dim=32
        )(concatenated, concatenated)
        
        # Add & normalize
        x = layers.Add()([concatenated, attention_output])
        x = layers.LayerNormalization()(x)
        
        # Feed-forward network
        ffn_output = layers.Dense(512, activation='swish')(x)
        ffn_output = layers.Dropout(0.1)(ffn_output)
        ffn_output = layers.Dense(256)(ffn_output)
        
        # Add & normalize again
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization()(x)
        
        # Global pooling to combine all features
        x = layers.GlobalAveragePooling1D()(x)
        
        return x
    
    def _transformer_encoder_block(self, x, heads=8, dim_head=64, dropout=0.1):
        """Transformer encoder block for spatial-temporal context"""
        # Reshape if needed
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=1)
        
        # Self-attention mechanism
        attention_output = layers.MultiHeadAttention(
            num_heads=heads, key_dim=dim_head, dropout=dropout
        )(x, x)
        
        # Add & normalize
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        ffn_output = layers.Dense(4 * x.shape[-1], activation='swish')(x)
        ffn_output = layers.Dropout(dropout)(ffn_output)
        ffn_output = layers.Dense(x.shape[-1])(ffn_output)
        ffn_output = layers.Dropout(dropout)(ffn_output)
        
        # Add & normalize again
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        return x
    
    def _cardiac_graph_neural_network(self, x):
        """Custom graph neural network for cardiac relationships"""
        # Simplified GNN implementation
        
        # Create node embeddings
        node_embeddings = layers.Dense(128, activation='swish')(x)
        
        # Simulate message passing with self-attention
        for _ in range(2):  # 2 message passing steps
            # Self-attention to simulate messages between nodes
            message = layers.Dense(128, activation='swish')(node_embeddings)
            message = layers.Dense(128)(message)
            
            # Update node embeddings
            node_embeddings = layers.Add()([node_embeddings, message])
            node_embeddings = layers.LayerNormalization()(node_embeddings)
        
        # Global pooling to aggregate node features
        graph_embedding = layers.Dense(256, activation='swish')(node_embeddings)
        
        return graph_embedding
    
    def _physiological_attention(self, x):
        """Advanced attention mechanism based on cardiac physiology regions in ECG images"""
        # Dynamic cardiac cycle-aware attention with learnable period parameters
        cycle_params = tf.Variable(initial_value=[0.2, 0.4, 0.6], trainable=True, name='cycle_params')
        
        # Generate attention scores for different physiological aspects
        attention_maps = []
        for wave_type in ['p_wave', 'qrs_complex', 't_wave', 'st_segment', 'qt_interval']:
            wave_features = layers.Dense(64, activation='swish', name=f'{wave_type}_attention')(x)
            wave_attention = layers.Dense(1, activation='sigmoid')(wave_features)
            attention_maps.append(wave_attention)
        
        # Dynamic fusion based on signal quality estimation
        quality_estimator = layers.Dense(len(attention_maps), activation='softmax', name='quality_estimator')(x)
        quality_estimator = tf.reshape(quality_estimator, [-1, 1, len(attention_maps)])
        
        # Weight attention maps by estimated quality
        attention_tensor = tf.stack(attention_maps, axis=-1)
        adaptive_attention = tf.reduce_sum(attention_tensor * quality_estimator, axis=-1)
        
        # Apply attention to features
        attended_features = layers.Multiply()([x, adaptive_attention])
        
        # Add residual connection for stable training
        attended_features = layers.Add()([x, attended_features])
        attended_features = layers.LayerNormalization()(attended_features)
        
        return attended_features
    
    def _meta_learning_adaptation(self, x):
        """Meta-learning module for patient-specific adaptation"""
        # Context vector generation
        context = layers.Dense(256, activation='swish')(x)
        
        # Hypernetwork to generate adaptive weights
        adaptive_weights = layers.Dense(128, activation='swish')(context)
        adaptive_weights = layers.Dense(128 * 64)(adaptive_weights)
        adaptive_weights = layers.Reshape((128, 64))(adaptive_weights)
        
        # Apply adaptive transformation
        features = layers.Dense(128, activation='swish')(x)
        features = tf.expand_dims(features, axis=-1)  # Add dimension for matmul
        
        # Matrix multiplication for dynamic adaptation
        adapted_features = tf.matmul(adaptive_weights, features)
        adapted_features = tf.squeeze(adapted_features, axis=-1)
        
        return adapted_features
    
    def _temporal_residual_block(self, x, filters, kernel_size):
        """Custom residual block for temporal processing"""
        # First Conv layer
        shortcut = x
        
        x = SpectralNormalization(Conv1D(filters, kernel_size, padding='same'))(x)
        x = BatchNormalization()(x)
        x = Activation('swish')(x)
        
        # Second Conv layer
        x = SpectralNormalization(Conv1D(filters, kernel_size, padding='same'))(x)
        x = BatchNormalization()(x)
        
        # Shortcut connection
        if shortcut.shape[-1] != filters:
            shortcut = Conv1D(filters, 1, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
        
        # Add shortcut to output
        x = Add()([x, shortcut])
        x = Activation('swish')(x)
        
        return x
    
    def _temporal_attention_module(self, x):
        """Temporal attention specifically for ECG signals"""
        # Generate attention weights
        attention = Conv1D(1, 7, padding='same')(x)
        attention = BatchNormalization()(attention)
        attention = Activation('sigmoid')(attention)
        
        # Apply attention
        return layers.Multiply()([x, attention])

    def _cardiac_cycle_detector(self, x):
        """Specialized module to identify cardiac cycle components"""
        # QRS detection submodule
        qrs_detector = Conv1D(32, 3, padding='same', activation='relu')(x)
        qrs_detector = Conv1D(1, 1, padding='same', activation='sigmoid')(qrs_detector)
        
        # P-wave detection submodule
        p_detector = Conv1D(32, 5, padding='same', activation='relu')(x)
        p_detector = Conv1D(1, 1, padding='same', activation='sigmoid')(p_detector)
        
        # T-wave detection submodule
        t_detector = Conv1D(32, 7, padding='same', activation='relu')(x)
        t_detector = Conv1D(1, 1, padding='same', activation='sigmoid')(t_detector)
        
        # Combine detectors with original features
        detectors = layers.Concatenate()([qrs_detector, p_detector, t_detector])
        enhanced = layers.Concatenate()([x, detectors])
        
        return enhanced
    
    def _wave_feature_extractor(self, x, wave_type):
        """Extract features specific to particular ECG waves"""
        # Different kernel sizes based on wave type
        if wave_type == 'p_wave':
            kernel_size = 9
        elif wave_type == 'qrs':
            kernel_size = 5
        elif wave_type == 't_wave':
            kernel_size = 11
        else:
            kernel_size = 7
            
        # Specialized convolution for this wave type
        x = Conv1D(32, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('swish')(x)
        
        # Extract morphological features
        x = Conv1D(64, kernel_size - 2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('swish')(x)
        
        # Global feature aggregation
        x = layers.GlobalMaxPooling1D()(x)
        
        return x
    
    def _contrastive_loss_fn(self, temperature=0.1):
        """Contrastive loss function for self-supervised learning"""
        def contrastive_loss(y_true, y_pred):
            # y_true is not used but required by Keras
            # y_pred contains [original_embeddings, augmented_embeddings]
            original_embeddings, augmented_embeddings = y_pred
            
            # Cosine similarity matrix
            batch_size = tf.shape(original_embeddings)[0]
            
            # Positive pairs - diagonal elements
            positive_similarity = tf.reduce_sum(
                original_embeddings * augmented_embeddings, axis=1
            )
            
            # All similarities
            similarities = tf.matmul(
                original_embeddings, tf.transpose(augmented_embeddings)
            ) / temperature
            
            # Create labels - diagonal elements are positive pairs
            labels = tf.eye(batch_size)
            
            # Compute loss (NT-Xent loss)
            loss = tf.keras.losses.categorical_crossentropy(
                y_true=labels,
                y_pred=tf.nn.softmax(similarities, axis=1),
                from_logits=False
            )
            
            return tf.reduce_mean(loss)
            
        return contrastive_loss
        
    def _adaptive_focal_loss(self, y_true, y_pred):
        """Specialized focal loss for imbalanced ECG datasets"""
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calculate focal weight
        alpha = 0.25
        gamma = 2.0
        focal_weight = alpha * tf.pow(1 - y_pred, gamma) * y_true
        
        # Add disease-specific weights (could be expanded)
        # Here we're simulating class weights
        class_weights = tf.constant([1.0, 2.0, 3.0, 2.5, 1.5], dtype=tf.float32)
        class_weights = tf.reshape(class_weights, [1, self.num_classes])
        weighted_loss = cross_entropy * focal_weight * class_weights
        
        return tf.reduce_sum(weighted_loss, axis=-1)
    
    def _build_optimizer(self):
        """Custom optimizer with learning rate schedule"""
        # Cosine decay learning rate schedule with warmup
        lr_schedule = self._cosine_decay_with_warmup(
            initial_learning_rate=self.learning_rate,
            decay_steps=10000,
            warmup_steps=1000,
            alpha=0.1
        )
        
        # AdamW optimizer with weight decay
        optimizer = tfa.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=self.weight_decay
        )
        
        return optimizer
        
    def _cosine_decay_with_warmup(self, initial_learning_rate, decay_steps, warmup_steps, alpha=0.0):
        """Custom learning rate schedule with warmup period"""
        def learning_rate_fn(step):
            # Convert to float
            step = tf.cast(step, tf.float32)
            decay_steps_float = tf.cast(decay_steps, tf.float32)
            warmup_steps_float = tf.cast(warmup_steps, tf.float32)
            
            # Warmup phase
            warmup_learning_rate = initial_learning_rate * (step / warmup_steps_float)
            
            # Cosine decay phase
            cosine_decay = 0.5 * (1 + tf.cos(
                3.14159265359 * (step - warmup_steps_float) / (decay_steps_float - warmup_steps_float)
            ))
            decayed = (1 - alpha) * cosine_decay + alpha
            cosine_learning_rate = initial_learning_rate * decayed
            
            # Use warmup_learning_rate if step < warmup_steps, else use cosine_learning_rate
            return tf.cond(
                step < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: cosine_learning_rate
            )
            
        return learning_rate_fn
    
    def _sensitivity(self, y_true, y_pred):
        """Calculate sensitivity/recall"""
        # For multi-class scenarios, treat each class separately and average
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        y_pred_one_hot = tf.one_hot(y_pred_classes, self.num_classes)
        
        true_positives = tf.reduce_sum(tf.cast(y_true * y_pred_one_hot, tf.float32), axis=0)
        possible_positives = tf.reduce_sum(tf.cast(y_true, tf.float32), axis=0)
        
        # Avoid division by zero
        class_sensitivity = true_positives / (possible_positives + tf.keras.backend.epsilon())
        
        # Average across classes (macro averaging)
        return tf.reduce_mean(class_sensitivity)
    
    def _specificity(self, y_true, y_pred):
        """Calculate specificity"""
        # For multi-class scenarios, treat each class separately and average
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        y_pred_one_hot = tf.one_hot(y_pred_classes, self.num_classes)
        
        true_negatives = tf.reduce_sum(tf.cast((1-y_true) * (1-y_pred_one_hot), tf.float32), axis=0)
        possible_negatives = tf.reduce_sum(tf.cast(1-y_true, tf.float32), axis=0)
        
        # Avoid division by zero
        class_specificity = true_negatives / (possible_negatives + tf.keras.backend.epsilon())
        
        # Average across classes (macro averaging)
        return tf.reduce_mean(class_specificity)
    
    def _f1_score(self, y_true, y_pred):
        """Calculate F1 score"""
        precision = self._precision(y_true, y_pred)
        recall = self._sensitivity(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
    
    def _precision(self, y_true, y_pred):
        """Calculate precision"""
        # For multi-class scenarios, treat each class separately and average
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        y_pred_one_hot = tf.one_hot(y_pred_classes, self.num_classes)
        
        true_positives = tf.reduce_sum(tf.cast(y_true * y_pred_one_hot, tf.float32), axis=0)
        predicted_positives = tf.reduce_sum(tf.cast(y_pred_one_hot, tf.float32), axis=0)
        
        # Avoid division by zero
        class_precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        
        # Average across classes (macro averaging)
        return tf.reduce_mean(class_precision)
            
    def fit(self, x_train, y_train, hrv_train=None, qt_train=None, 
            validation_data=None, validation_hrv=None, validation_qt=None,
            epochs=100, batch_size=32, class_weights=None):
        """
        Train the CardioVisionNet model with multi-task learning
        
        Args:
            x_train: ECG input data
            y_train: CVD classification targets
            hrv_train: HRV regression targets (optional)
            qt_train: QT interval regression targets (optional)
            validation_data: Tuple of (x_val, y_val) for validation
            validation_hrv: HRV validation targets
            validation_qt: QT interval validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            class_weights: Optional dictionary mapping class indices to weights
            
        Returns:
            Training history
        """
        # Setup training data based on available targets
        if hrv_train is None:
            hrv_train = np.zeros((len(x_train), 1))  # Dummy data
            
        if qt_train is None:
            qt_train = np.zeros((len(x_train), 1))  # Dummy data
            
        # Setup validation data
        val_hrv = None
        val_qt = None
        
        if validation_data is not None:
            x_val, y_val = validation_data
            
            if validation_hrv is None:
                val_hrv = np.zeros((len(x_val), 1))
            else:
                val_hrv = validation_hrv
                
            if validation_qt is None:
                val_qt = np.zeros((len(x_val), 1))
            else:
                val_qt = validation_qt
        
        # Create dummy data for component weights and component predictions
        # We don't have ground truth for these, so they're for regularization purposes
        component_names = ['p_wave', 'qrs_complex', 't_wave', 'st_segment', 'qt_interval']
        n_components = len(component_names)
        
        # Create uniform distribution for component weights (no prior knowledge)
        uniform_weights = np.ones((len(x_train), n_components)) / n_components
        
        # Setup training data with all outputs
        y_dict = {
            "cvd_prediction": y_train,
            "uncertainty": np.zeros_like(y_train),  # Typically unsupervised
            "hrv_prediction": hrv_train,
            "gradcam_output": np.zeros((len(x_train), 1)),  # Dummy data
            "component_weights": uniform_weights
        }
        
        # Add component-specific targets (same as main target for now)
        for component in component_names:
            y_dict[f"{component}_prediction"] = y_train
        
        # Setup validation data with all outputs if available
        validation_data_dict = None
        if validation_data is not None:
            val_uniform_weights = np.ones((len(x_val), n_components)) / n_components
            val_dict = {
                "cvd_prediction": y_val,
                "uncertainty": np.zeros_like(y_val),
                "hrv_prediction": val_hrv,
                "gradcam_output": np.zeros((len(x_val), 1)),
                "component_weights": val_uniform_weights
            }
            
            # Add component-specific validation targets
            for component in component_names:
                val_dict[f"{component}_prediction"] = y_val
                
            validation_data_dict = (x_val, val_dict)
        
        # Setup callbacks
        log_dir = os.path.join(self.model_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        callbacks = [
            EarlyStopping(monitor='val_cvd_prediction_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_cvd_prediction_loss', factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(
                filepath=os.path.join(self.model_dir, 'best_model.h5'),
                monitor='val_cvd_prediction_loss',
                save_best_only=True,
                save_weights_only=True
            ),
            TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='500,520')
        ]
        
        # Train model
        return self.model.fit(
            x_train, y_dict,
            validation_data=validation_data_dict,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights
        )

    def predict(self, x, monte_carlo_samples=10):
        """
        Predict with uncertainty estimation using Monte Carlo dropout
        
        Args:
            x: Input ECG data
            monte_carlo_samples: Number of forward passes with dropout enabled
            
        Returns:
            Dictionary containing:
                predictions: Mean prediction probabilities
                uncertainties: Standard deviation of predictions
                hrv_predictions: Heart rate variability predictions
                component_predictions: Component-specific predictions
                component_weights: Importance weights for each cardiac component
                all_samples: All Monte Carlo samples (if needed for further analysis)
        """
        # Enable dropout during inference
        cvd_predictions = []
        uncertainty_estimates = []
        hrv_predictions = []
        gradcam_outputs = []
        component_weights_list = []
        
        # Track component-specific predictions
        component_names = ['p_wave', 'qrs_complex', 't_wave', 'st_segment', 'qt_interval']
        component_predictions = {name: [] for name in component_names}
        
        # Multiple forward passes with dropout enabled
        for _ in range(monte_carlo_samples):
            outputs = self.model(x, training=True)
            
            # Extract different outputs - order matches the model's outputs list
            cvd_pred = outputs[0]
            uncertainty = outputs[1]
            hrv_pred = outputs[2]
            gradcam = outputs[3]
            comp_weights = outputs[4]
            
            # Store main predictions
            cvd_predictions.append(cvd_pred)
            uncertainty_estimates.append(uncertainty)
            hrv_predictions.append(hrv_pred)
            gradcam_outputs.append(gradcam)
            component_weights_list.append(comp_weights)
            
            # Store component-specific predictions
            for i, component in enumerate(component_names):
                component_predictions[component].append(outputs[5 + i])
            
        # Stack predictions
        cvd_preds_stacked = tf.stack(cvd_predictions, axis=0)
        uncertainty_stacked = tf.stack(uncertainty_estimates, axis=0)
        hrv_preds_stacked = tf.stack(hrv_predictions, axis=0)
        component_weights_stacked = tf.stack(component_weights_list, axis=0)
        
        # Stack component predictions
        component_preds_stacked = {
            name: tf.stack(preds, axis=0) for name, preds in component_predictions.items()
        }
        
        # Calculate mean and standard deviation for CVD predictions
        mean_cvd_pred = tf.reduce_mean(cvd_preds_stacked, axis=0)
        std_cvd_pred = tf.math.reduce_std(cvd_preds_stacked, axis=0)
        
        # Calculate mean for other outputs
        mean_uncertainty = tf.reduce_mean(uncertainty_stacked, axis=0)
        mean_hrv = tf.reduce_mean(hrv_preds_stacked, axis=0)
        mean_component_weights = tf.reduce_mean(component_weights_stacked, axis=0)
        
        # Calculate mean for component predictions
        mean_component_preds = {
            name: tf.reduce_mean(preds, axis=0) for name, preds in component_preds_stacked.items()
        }
        
        # Return all predictions
        return {
            "cvd_predictions": mean_cvd_pred,
            "cvd_uncertainties": std_cvd_pred,
            "model_uncertainty": mean_uncertainty,
            "hrv_predictions": mean_hrv,
            "component_weights": mean_component_weights,
            "component_predictions": mean_component_preds,
            "all_cvd_samples": cvd_preds_stacked,  # For further uncertainty analysis
            "all_component_samples": component_preds_stacked  # For component-specific uncertainty
        }
        
    def evaluate_model(self, x_test, y_test, hrv_test=None, qt_test=None, 
                     monte_carlo_samples=10, verbose=1):
        """
        Comprehensive model evaluation with uncertainty estimation and explainability
        
        Args:
            x_test: Test ECG data
            y_test: Test labels (one-hot encoded)
            hrv_test: Heart rate variability ground truth (optional)
            qt_test: QT interval ground truth (optional)
            monte_carlo_samples: Number of MC samples for uncertainty
            verbose: Verbosity level
            
        Returns:
            Dictionary with evaluation metrics and explainability information
        """
        # Get predictions with uncertainty
        predictions = self.predict(x_test, monte_carlo_samples=monte_carlo_samples)
        
        # Get class predictions
        y_pred = predictions['cvd_predictions']
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Calculate standard metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import confusion_matrix, classification_report
        from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
        
        # Basic metrics
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        precision_macro = precision_score(y_true_classes, y_pred_classes, average='macro')
        recall_macro = recall_score(y_true_classes, y_pred_classes, average='macro')
        f1_macro = f1_score(y_true_classes, y_pred_classes, average='macro')
        
        # Class-wise metrics
        precision_class = precision_score(y_true_classes, y_pred_classes, average=None)
        recall_class = recall_score(y_true_classes, y_pred_classes, average=None)
        f1_class = f1_score(y_true_classes, y_pred_classes, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        # Calculate ROC AUC for each class
        roc_auc = {}
        for i in range(self.num_classes):
            roc_auc[f'class_{i}'] = roc_auc_score(y_test[:, i], y_pred[:, i])
            
        # Calculate average AUC
        avg_auc = np.mean(list(roc_auc.values()))
        
        # Get uncertainty metrics
        uncertainties = predictions['cvd_uncertainties']
        model_uncertainty = predictions['model_uncertainty']
        
        # Calculate correlation between uncertainty and error
        correct_pred = (y_pred_classes == y_true_classes)
        error_uncertainty_corr = np.corrcoef(~correct_pred, np.mean(uncertainties, axis=1))[0, 1]
        
        # Calculate other HRV and QT metrics if available
        hrv_metrics = {}
        qt_metrics = {}
        
        if hrv_test is not None:
            hrv_pred = predictions['hrv_predictions']
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            hrv_metrics = {
                'mae': mean_absolute_error(hrv_test, hrv_pred),
                'rmse': np.sqrt(mean_squared_error(hrv_test, hrv_pred)),
                'r2': r2_score(hrv_test, hrv_pred)
            }
            
        if qt_test is not None:
            qt_pred = predictions['qt_predictions']
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            qt_metrics = {
                'mae': mean_absolute_error(qt_test, qt_pred),
                'rmse': np.sqrt(mean_squared_error(qt_test, qt_pred)),
                'r2': r2_score(qt_test, qt_pred)
            }
            
        # Classification report
        class_report = classification_report(y_true_classes, y_pred_classes, output_dict=True)
        
        if verbose:
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision (macro): {precision_macro:.4f}")
            print(f"Recall (macro): {recall_macro:.4f}")
            print(f"F1 Score (macro): {f1_macro:.4f}")
            print(f"ROC AUC (avg): {avg_auc:.4f}")
            print("\nConfusion Matrix:")
            print(cm)
            print("\nClassification Report:")
            print(classification_report(y_true_classes, y_pred_classes))
            print(f"\nUncertainty-Error Correlation: {error_uncertainty_corr:.4f}")
            
            if hrv_metrics:
                print("\nHRV Metrics:")
                for key, value in hrv_metrics.items():
                    print(f"{key}: {value:.4f}")
                    
            if qt_metrics:
                print("\nQT Interval Metrics:")
                for key, value in qt_metrics.items():
                    print(f"{key}: {value:.4f}")
            
        # Combine all metrics
        results = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_class': precision_class,
            'recall_class': recall_class,
            'f1_class': f1_class,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'avg_auc': avg_auc,
            'error_uncertainty_corr': error_uncertainty_corr,
            'classification_report': class_report,
            'predictions': y_pred,
            'uncertainties': uncertainties,
            'model_uncertainty': model_uncertainty,
            'hrv_metrics': hrv_metrics,
            'qt_metrics': qt_metrics,
            'correct_predictions': correct_pred,
            'y_true': y_test
        }
        
        return results
        
    def plot_results(self, evaluation_results, save_dir=None):
        """
        Generate visualizations for model evaluation with explainability
        
        Args:
            evaluation_results: Results from evaluate_model method
            save_dir: Directory to save plots (None = display only)
            
        Returns:
            Dictionary with figure objects
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        figures = {}
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = evaluation_results['confusion_matrix']
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix')
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        
        figures['confusion_matrix'] = plt.gcf()
        
        # 2. ROC Curves
        plt.figure(figsize=(10, 8))
        
        y_true = evaluation_results.get('y_true')
        y_pred = evaluation_results['predictions']
        
        if y_true is not None:
            for i in range(self.num_classes):
                RocCurveDisplay.from_predictions(
                    y_true[:, i],
                    y_pred[:, i],
                    name=f'Class {i}',
                    alpha=0.7
                )
        
        plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
            
        figures['roc_curves'] = plt.gcf()
        
        # 3. Uncertainty Distribution
        plt.figure(figsize=(10, 6))
        
        uncertainties = evaluation_results['uncertainties']
        mean_uncertainties = np.mean(uncertainties, axis=1)
        
        correct_pred = evaluation_results.get('correct_predictions')
        if correct_pred is not None:
            plt.hist(mean_uncertainties[correct_pred], bins=20, alpha=0.5, label='Correct Predictions')
            plt.hist(mean_uncertainties[~correct_pred], bins=20, alpha=0.5, label='Incorrect Predictions')
            plt.legend()
        else:
            plt.hist(mean_uncertainties, bins=20)
            
        plt.xlabel('Model Uncertainty')
        plt.ylabel('Count')
        plt.title('Uncertainty Distribution')
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'uncertainty_dist.png'), dpi=300, bbox_inches='tight')
            
        figures['uncertainty_dist'] = plt.gcf()
        
        # 4. Class-wise Performance
        plt.figure(figsize=(12, 6))
        
        metrics_class = {
            'Precision': evaluation_results['precision_class'],
            'Recall': evaluation_results['recall_class'],
            'F1 Score': evaluation_results['f1_class']
        }
        
        x = np.arange(len(metrics_class['Precision']))
        width = 0.25
        
        plt.bar(x - width, metrics_class['Precision'], width, label='Precision')
        plt.bar(x, metrics_class['Recall'], width, label='Recall')
        plt.bar(x + width, metrics_class['F1 Score'], width, label='F1 Score')
        
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Class-wise Performance Metrics')
        plt.xticks(x, [f'Class {i}' for i in range(len(x))])
        plt.legend()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'class_performance.png'), dpi=300, bbox_inches='tight')
            
        figures['class_performance'] = plt.gcf()
        
        # 5. NEW: Component Importance Visualization
        if 'component_importance' in evaluation_results:
            plt.figure(figsize=(10, 6))
            
            component_importance = evaluation_results['component_importance']
            components = list(component_importance.keys())
            importances = [component_importance[c] for c in components]
            
            # Sort by importance
            sorted_idx = np.argsort(importances)
            sorted_components = [components[i] for i in sorted_idx]
            sorted_importances = [importances[i] for i in sorted_idx]
            
            # Format component names for better display
            display_components = [c.replace('_', ' ').title() for c in sorted_components]
            
            plt.barh(display_components, sorted_importances, color='skyblue')
            plt.xlabel('Importance Weight')
            plt.ylabel('ECG Component')
            plt.title('Cardiac Component Importance')
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'component_importance.png'), dpi=300, bbox_inches='tight')
                
            figures['component_importance'] = plt.gcf()
        
        # 6. NEW: Component Performance Comparison
        if 'component_metrics' in evaluation_results:
            plt.figure(figsize=(12, 6))
            
            component_metrics = evaluation_results['component_metrics']
            components = list(component_metrics.keys())
            
            # Extract accuracy and F1 for each component
            accuracies = [component_metrics[c]['accuracy'] for c in components]
            f1_scores = [component_metrics[c]['f1_score'] for c in components]
            
            # Add overall model performance
            components.append('Full Model')
            accuracies.append(evaluation_results['accuracy'])
            f1_scores.append(evaluation_results['f1_macro'])
            
            # Format component names for better display
            display_components = [c.replace('_', ' ').title() for c in components]
            
            x = np.arange(len(components))
            width = 0.35
            
            plt.bar(x - width/2, accuracies, width, label='Accuracy')
            plt.bar(x + width/2, f1_scores, width, label='F1 Score')
            
            plt.xlabel('Component')
            plt.ylabel('Score')
            plt.title('Performance by Cardiac Component')
            plt.xticks(x, display_components, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'component_performance.png'), dpi=300, bbox_inches='tight')
                
            figures['component_performance'] = plt.gcf()
        
        # 7. NEW: Class-Component Importance Heatmap
        if 'class_components_importance' in evaluation_results:
            plt.figure(figsize=(12, 8))
            
            class_components = evaluation_results['class_components_importance']
            
            # Create a matrix for the heatmap
            classes = list(class_components.keys())
            if len(classes) > 0:
                first_class = list(class_components.keys())[0]
                components = list(class_components[first_class].keys())
                
                heatmap_data = np.zeros((len(classes), len(components)))
                
                for i, cls in enumerate(classes):
                    for j, comp in enumerate(components):
                        heatmap_data[i, j] = class_components[cls][comp]
                
                # Format axis labels
                display_components = [c.replace('_', ' ').title() for c in components]
                display_classes = [c.replace('_', ' ').title() for c in classes]
                
                # Create heatmap
                ax = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu",
                               xticklabels=display_components, yticklabels=display_classes)
                plt.title('Component Importance by Class')
                plt.xlabel('Cardiac Component')
                plt.ylabel('Class')
                plt.tight_layout()
                
                if save_dir:
                    plt.savefig(os.path.join(save_dir, 'class_component_heatmap.png'), dpi=300, bbox_inches='tight')
                    
                figures['class_component_heatmap'] = plt.gcf()
        
        # 8. NEW: Component Error Analysis
        if 'component_error_analysis' in evaluation_results:
            plt.figure(figsize=(10, 6))
            
            component_error = evaluation_results['component_error_analysis']
            components = list(component_error.keys())
            error_diffs = [component_error[c] for c in components]
            
            # Sort by absolute difference
            sorted_idx = np.argsort([abs(d) for d in error_diffs])[::-1]  # Reverse for descending
            sorted_components = [components[i] for i in sorted_idx]
            sorted_diffs = [error_diffs[i] for i in sorted_idx]
            
            # Format component names for better display
            display_components = [c.replace('_', ' ').title() for c in sorted_components]
            
            # Use different colors for positive and negative differences
            colors = ['green' if d > 0 else 'red' for d in sorted_diffs]
            
            plt.barh(display_components, sorted_diffs, color=colors)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel('Importance Difference (Correct - Incorrect)')
            plt.ylabel('ECG Component')
            plt.title('Component Error Analysis\nPositive values indicate greater importance in correct predictions')
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'component_error_analysis.png'), dpi=300, bbox_inches='tight')
                
            figures['component_error_analysis'] = plt.gcf()
        
        # Return all figure objects
        return figures

    def save(self, filepath=None):
        """Save the model weights and configuration"""
        if filepath is None:
            filepath = os.path.join(self.model_dir, "cardiovisionnet_model")
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save weights
        self.model.save_weights(f"{filepath}.h5")
        
        # Save model configuration
        config = {
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dropout_rate": self.dropout_rate,
            "filters_base": self.filters_base,
            "use_self_supervision": self.use_self_supervision,
            "use_attention": self.use_attention,
            "backbone": self.backbone
        }
        
        import json
        with open(f"{filepath}_config.json", 'w') as f:
            json.dump(config, f)
            
        print(f"Model saved to {filepath}")
        
    @classmethod
    def load(cls, filepath):
        """Load a model from weights and configuration"""
        # Load configuration
        import json
        with open(f"{filepath}_config.json", 'r') as f:
            config = json.load(f)
            
        # Create model with loaded config
        model = cls(**config)
        
        # Load weights
        model.model.load_weights(f"{filepath}.h5")
        
        print(f"Model loaded from {filepath}")
        return model

    def _image_preprocessing_module(self, inputs):
        """Advanced image preprocessing and enhancement for ECG charts"""
        # Normalization
        x = layers.Lambda(lambda x: x / 255.0)(inputs)
        
        # Contrast enhancement with learnable parameters
        alpha = tf.Variable(1.0, trainable=True)
        beta = tf.Variable(0.0, trainable=True)
        x = layers.Lambda(lambda x: alpha * x + beta)(x)
        
        # Edge enhancement with convolutions
        edge_detect = layers.Conv2D(16, 3, padding='same', activation='linear')(x)
        edge_detect = layers.BatchNormalization()(edge_detect)
        edge_detect = layers.Activation('tanh')(edge_detect)
        
        # Combine with original
        x = layers.Concatenate()([x, edge_detect])
        x = layers.Conv2D(self.input_shape[-1], 1, padding='same')(x)
        
        # Apply learnable gamma correction for better feature visibility
        gamma = tf.Variable(1.0, trainable=True)
        x = layers.Lambda(lambda x: tf.pow(x, gamma))(x)
        
        return x
    
    def _cnn_backbone_pathway(self, x):
        """Main CNN backbone for feature extraction from ECG images"""
        # Choose backbone based on configuration
        if self.backbone == 'efficientnet':
            base_model = EfficientNetV2B0(
                include_top=False, 
                weights='imagenet',
                input_shape=self.input_shape,
                drop_connect_rate=0.2
            )
        elif self.backbone == 'resnet':
            base_model = ResNet50(
                include_top=False,
                weights='imagenet',
                input_shape=self.input_shape
            )
        elif self.backbone == 'densenet':
            base_model = DenseNet121(
                include_top=False,
                weights='imagenet',
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unknown backbone: {self.backbone}")
        
        # Freeze early layers for transfer learning
        for layer in base_model.layers[:int(len(base_model.layers) * 0.7)]:
            layer.trainable = False
            
        # Get backbone features
        backbone_features = base_model(x)
        
        # Add ECG-specific adaptation layers
        adapted_features = layers.Conv2D(self.filters_base * 4, 1, padding='same')(backbone_features)
        adapted_features = layers.BatchNormalization()(adapted_features)
        adapted_features = layers.Activation('swish')(adapted_features)
        
        # Global pooling
        pooled_features = layers.GlobalAveragePooling2D()(adapted_features)
        
        return pooled_features
    
    def _local_pattern_pathway(self, x):
        """Extract local ECG patterns like P-QRS-T waves from images"""
        # Multi-scale pattern detection
        # Small receptive field for fine details (P, Q, S waves)
        small_patterns = layers.Conv2D(self.filters_base, 3, strides=1, padding='same')(x)
        small_patterns = layers.BatchNormalization()(small_patterns)
        small_patterns = layers.Activation('relu')(small_patterns)
        small_patterns = layers.MaxPooling2D(2)(small_patterns)
        
        # Medium receptive field for QRS complexes
        medium_patterns = layers.Conv2D(self.filters_base, 5, strides=1, padding='same')(x)
        medium_patterns = layers.BatchNormalization()(medium_patterns)
        medium_patterns = layers.Activation('relu')(medium_patterns)
        medium_patterns = layers.MaxPooling2D(2)(medium_patterns)
        
        # Large receptive field for T waves and overall morphology
        large_patterns = layers.Conv2D(self.filters_base, 9, strides=1, padding='same')(x)
        large_patterns = layers.BatchNormalization()(large_patterns)
        large_patterns = layers.Activation('relu')(large_patterns)
        large_patterns = layers.MaxPooling2D(2)(large_patterns)
        
        # Combine patterns at same resolution
        small_patterns = layers.Conv2D(self.filters_base, 3, strides=1, padding='same')(small_patterns)
        medium_patterns = layers.Conv2D(self.filters_base, 3, strides=1, padding='same')(medium_patterns)
        large_patterns = layers.Conv2D(self.filters_base, 3, strides=1, padding='same')(large_patterns)
        
        # Deep feature extraction
        small_patterns = self._ecg_pattern_block(small_patterns, self.filters_base*2)
        medium_patterns = self._ecg_pattern_block(medium_patterns, self.filters_base*2)
        large_patterns = self._ecg_pattern_block(large_patterns, self.filters_base*2)
        
        # Global pooling for each scale
        small_pool = layers.GlobalAveragePooling2D()(small_patterns)
        medium_pool = layers.GlobalAveragePooling2D()(medium_patterns)
        large_pool = layers.GlobalAveragePooling2D()(large_patterns)
        
        # Combine all scales
        combined = layers.Concatenate()([small_pool, medium_pool, large_pool])
        
        # Feature integration
        local_features = layers.Dense(256, activation='swish')(combined)
        
        return local_features
    
    def _wavelet_transform_pathway(self, x):
        """Advanced learnable wavelet transform for extracting frequency and scale features"""
        # Convert to grayscale if RGB
        if self.input_shape[-1] == 3:
            x_gray = layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x))(x)
        else:
            x_gray = x
        
        # Create learnable wavelet parameters - mother wavelet shape
        wavelet_size = 9
        mother_wavelet = tf.Variable(
            initial_value=tf.random.normal([wavelet_size, wavelet_size, 1, 16]),
            trainable=True, name='mother_wavelet'
        )
        
        # Learnable scales parameters
        scales = tf.Variable(
            initial_value=[1.0, 2.0, 4.0, 8.0, 16.0],
            trainable=True, name='wavelet_scales'
        )
        
        # Function to create scaled versions of the mother wavelet
        def create_scaled_wavelet(scale_idx):
            scale = scales[scale_idx]
            # Dynamically resize the wavelet kernel based on scale
            scaled_size = tf.cast(tf.math.ceil(wavelet_size * scale), tf.int32)
            # Ensure odd size for symmetric wavelet
            scaled_size = scaled_size + (1 - scaled_size % 2)
            # Resize the mother wavelet
            scaled_wavelet = tf.image.resize(
                mother_wavelet[:,:,0,:], 
                [scaled_size, scaled_size],
                method='bilinear'
            )
            # Reshape for conv2d
            scaled_wavelet = tf.reshape(scaled_wavelet, 
                                       [scaled_size, scaled_size, 1, 16])
            return scaled_wavelet
            
        # Apply multi-scale wavelet transform
        wavelet_outputs = []
        for i in range(len(scales)):
            # Get scaled wavelet
            wavelet_kernel = create_scaled_wavelet(i)
            # Apply wavelet as convolution
            wavelet_response = tf.nn.conv2d(
                x_gray, wavelet_kernel, strides=[1,1,1,1], padding='SAME'
            )
            wavelet_outputs.append(wavelet_response)
        
        # Process each scale
        processed_scales = []
        for i, wavelet_output in enumerate(wavelet_outputs):
            # Further process each scale
            scale_features = layers.Conv2D(32, 3, padding='same', activation='relu')(wavelet_output)
            scale_features = layers.BatchNormalization()(scale_features)
            # Add frequency-domain regularization
            scale_features = self._add_frequency_constraints(scale_features, i)
            
            # Global attention to identify important regions in this scale
            attention = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(scale_features)
            attended_features = layers.Multiply()([scale_features, attention])
            
            # Downsample proportionally to scale
            pool_size = 2 ** min(i, 3)  # Limit max pooling size
            if pool_size > 1:
                attended_features = layers.MaxPooling2D(pool_size)(attended_features)
                
            processed_scales.append(attended_features)
        
        # Adaptive cross-scale fusion
        # First normalize all outputs to same spatial dimensions
        resized_scales = []
        target_shape = processed_scales[-1].shape[1:3]  # Use smallest scale's shape
        
        for scale_features in processed_scales:
            # Resize to match target shape
            if scale_features.shape[1:3] != target_shape:
                resized = layers.Reshape(
                    (scale_features.shape[1], scale_features.shape[2], -1)
                )(scale_features)
                resized = tf.image.resize(
                    resized, target_shape, method='bilinear'
                )
            else:
                resized = scale_features
            resized_scales.append(resized)
        
        # Concatenate all scales along channel dimension
        multi_scale_features = layers.Concatenate(axis=-1)(resized_scales)
        
        # Channel attention for scale importance weighting
        scale_importance = self._channel_attention_module(multi_scale_features)
        weighted_features = layers.Multiply()([multi_scale_features, scale_importance])
        
        # Add 1x1 conv to reduce channels and integrate information
        integrated_features = layers.Conv2D(64, 1, padding='same', activation='swish')(weighted_features)
        integrated_features = layers.BatchNormalization()(integrated_features)
        
        # Global pooling
        pooled_features = layers.GlobalAveragePooling2D()(integrated_features)
        
        # Final integration
        wavelet_features = layers.Dense(256, activation='swish')(pooled_features)
        
        return wavelet_features
        
    def _add_frequency_constraints(self, x, scale_idx):
        """Add frequency-domain constraints according to scale"""
        # Different constraints for different scales
        if scale_idx == 0:  # High frequency (small scale)
            # High-pass characteristic - encourage high frequency preservation
            return layers.Lambda(lambda x: x * 1.2)(x)
        elif scale_idx == len(scales) - 1:  # Low frequency (large scale)
            # Low-pass characteristic - smooth activations
            smoothing_kernel = tf.constant(
                [[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]],
                dtype=tf.float32
            )
            smoothing_kernel = tf.reshape(smoothing_kernel, [3, 3, 1, 1])
            smoothing_kernel = tf.tile(smoothing_kernel, [1, 1, x.shape[-1], 1])
            
            # Apply smoothing
            x_smoothed = tf.nn.depthwise_conv2d(
                x, smoothing_kernel, strides=[1,1,1,1], padding='SAME'
            )
            return x_smoothed
        else:
            # Band-pass characteristic for mid-scales
            return x
            
    def _channel_attention_module(self, x):
        """Channel attention module for weighting different frequency bands"""
        # Global average pooling for each channel
        gap = layers.GlobalAveragePooling2D()(x)
        gap = tf.reshape(gap, [-1, 1, 1, x.shape[-1]])
        
        # Global max pooling for each channel
        gmp = layers.GlobalMaxPooling2D()(x)
        gmp = tf.reshape(gmp, [-1, 1, 1, x.shape[-1]])
        
        # Combine pooling operations
        combined = layers.Concatenate(axis=-1)([gap, gmp])
        
        # Shared MLP
        mlp = layers.Conv2D(x.shape[-1] // 4, 1, activation='relu')(combined)
        mlp = layers.Conv2D(x.shape[-1], 1, activation='sigmoid')(mlp)
        
        # Split the attention weights back
        attention = mlp[:, :, :, :x.shape[-1]]
        
        return attention
    
    def _ecg_pattern_block(self, x, filters):
        """Specialized convolutional block for extracting ECG patterns"""
        # Depthwise separable convolution for efficient feature extraction
        x = layers.SeparableConv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x)
        
        # Residual connection
        shortcut = x
        
        # Expand features
        x = layers.SeparableConv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x)
        
        x = layers.SeparableConv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Add shortcut
        x = layers.Add()([x, shortcut])
        x = layers.Activation('swish')(x)
        
        # Spatial attention to focus on important regions
        attention = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(x)
        x = layers.Multiply()([x, attention])
        
        return x

    def _decision_pathway_explainer(self, features):
        """Novel interpretability module that explains decisions through cardiac-specific pathways"""
        # Create explanation masks for each cardiac component
        explanation_components = {}
        explanation_logits = {}
        
        for component in ['p_wave', 'qrs_complex', 't_wave', 'st_segment', 'qt_interval']:
            # Create component-specific attention mask
            mask = layers.Dense(features.shape[-1], activation='sigmoid', name=f'{component}_mask')(features)
            # Apply mask to features to get component-specific contribution
            component_contribution = layers.Multiply()([features, mask])
            # Get logits for this component
            component_logits = layers.Dense(self.num_classes, name=f'{component}_logits')(component_contribution)
            # Store for later use
            explanation_components[component] = component_contribution
            explanation_logits[component] = component_logits
        
        # Evidential deep learning for uncertainty quantification
        # Generate evidence for Dirichlet distribution parameters
        evidence = layers.Dense(self.num_classes * 4, activation='softplus')(features)
        alpha, beta, gamma, delta = tf.split(evidence, 4, axis=-1)
        
        # Calculate uncertainty from evidential parameters
        # Epistemic (model) uncertainty from variance of the Dirichlet distribution
        uncertainty = beta / (alpha - 1 + 1e-10)  # Add epsilon to prevent division by zero
        
        # Calculate uncertainty for each component
        component_uncertainties = {}
        for component, comp_features in explanation_components.items():
            comp_evidence = layers.Dense(4, activation='softplus')(comp_features)
            comp_alpha = comp_evidence[:, 0:1]
            comp_uncertainty = 1.0 / (comp_alpha + 1e-10)
            component_uncertainties[component] = comp_uncertainty
        
        # Calculate final logits as weighted sum of component logits
        component_weights = layers.Dense(len(explanation_components), activation='softmax', 
                                        name='component_weights')(features)
        
        final_logits = None
        for i, (component, logits) in enumerate(explanation_logits.items()):
            weighted_logits = layers.Lambda(
                lambda x: x[0] * tf.reshape(x[1][:, i], [-1, 1])
            )([logits, component_weights])
            
            if final_logits is None:
                final_logits = weighted_logits
            else:
                final_logits = layers.Add()([final_logits, weighted_logits])
        
        return final_logits, uncertainty, explanation_logits, component_uncertainties, component_weights

#----------------------- Data Loading Functions -----------------------#

def load_data_from_folders(data_folder, labels_file=None, lead_count=12, sample_length=5000):
    """
    Load ECG data from a folder structure
    
    Args:
        data_folder: Path to folder containing ECG files (supports nested folders)
        labels_file: Optional path to CSV/Excel file with labels
        lead_count: Number of ECG leads to expect
        sample_length: Length to resample all ECG signals to
        
    Returns:
        X_data: ECG data as numpy array (samples, time_points, leads)
        y_labels: Labels as numpy array
    """
    print(f"Loading data from folder: {data_folder}")
    
    # Find all ECG files (supporting multiple formats)
    ecg_files = []
    
    # Support for .mat files (MATLAB)
    ecg_files.extend(glob.glob(os.path.join(data_folder, "**/*.mat"), recursive=True))
    
    # Support for .csv files
    ecg_files.extend(glob.glob(os.path.join(data_folder, "**/*.csv"), recursive=True))
    
    # Support for .txt files
    ecg_files.extend(glob.glob(os.path.join(data_folder, "**/*.txt"), recursive=True))
    
    # Support for .edf files (European Data Format for EEG/ECG)
    ecg_files.extend(glob.glob(os.path.join(data_folder, "**/*.edf"), recursive=True))
    
    print(f"Found {len(ecg_files)} ECG files")
    
    if len(ecg_files) == 0:
        raise ValueError(f"No supported ECG files found in {data_folder}")
    
    # Load labels if provided
    labels_dict = None
    if labels_file is not None:
        labels_dict = load_labels_file(labels_file)
    
    # Process each file and load the data
    X_data = []
    y_labels = []
    file_ids = []
    
    for file_path in ecg_files:
        try:
            # Extract file ID (using filename without extension)
            file_id = os.path.splitext(os.path.basename(file_path))[0]
            file_ids.append(file_id)
            
            # Load ECG data based on file format
            ecg_data = load_ecg_file(file_path, lead_count, sample_length)
            
            if ecg_data is not None:
                X_data.append(ecg_data)
                
                # Get label if available
                if labels_dict is not None:
                    if file_id in labels_dict:
                        y_labels.append(labels_dict[file_id])
                    else:
                        # If no label found, use a placeholder
                        y_labels.append(-1)  # Will be filtered later
                        print(f"Warning: No label found for {file_id}")
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    # Convert to numpy arrays
    X_data = np.array(X_data)
    
    # Handle labels
    if labels_dict is not None:
        y_labels = np.array(y_labels)
        
        # Filter out samples without labels
        if -1 in y_labels:
            valid_indices = y_labels != -1
            X_data = X_data[valid_indices]
            y_labels = y_labels[valid_indices]
            print(f"Filtered out {np.sum(~valid_indices)} samples without labels")
    else:
        # If no labels file provided, create dummy labels
        y_labels = np.zeros(len(X_data))
        print("No labels file provided. Created dummy labels.")
    
    print(f"Loaded {X_data.shape[0]} samples with shape {X_data.shape[1:]} and {len(np.unique(y_labels))} classes")
    
    return X_data, y_labels

def load_data_from_zip(zip_file, labels_file=None, lead_count=12, sample_length=5000):
    """Load ECG data from a zip file"""
    print(f"Loading data from zip file: {zip_file}")
    
    # Create a temporary directory for extraction
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Extract zip file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Load data from the extracted folder
        return load_data_from_folders(temp_dir, labels_file, lead_count, sample_length)
        
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)

def load_labels_file(labels_file):
    """Load labels from CSV or Excel file"""
    # Determine file type and load accordingly
    file_ext = os.path.splitext(labels_file)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(labels_file)
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(labels_file)
    else:
        raise ValueError(f"Unsupported labels file format: {file_ext}")
    
    # Try to find columns with similar names
    file_id_col = None
    label_col = None
    
    for col in df.columns:
        if col.lower() in ['file_id', 'fileid', 'id', 'filename', 'file']:
            file_id_col = col
        elif col.lower() in ['label', 'class', 'diagnosis', 'category', 'condition']:
            label_col = col
    
    if file_id_col is None or label_col is None:
        raise ValueError(f"Could not find required columns in labels file. Needed: file_id, label. Found: {df.columns.tolist()}")
    
    # Create dictionary mapping file IDs to labels
    labels_dict = {}
    
    for _, row in df.iterrows():
        file_id = str(row[file_id_col])
        label = row[label_col]
        labels_dict[file_id] = label
    
    print(f"Loaded {len(labels_dict)} labels from {labels_file}")
    
    return labels_dict

def load_ecg_file(file_path, lead_count=12, sample_length=5000):
    """Load ECG data from various file formats"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.mat':
            # For MATLAB files
            mat_data = loadmat(file_path)
            
            # Find the ECG data array - mat files can have different structures
            ecg_data = None
            
            # Common field names in ECG mat files
            possible_fields = ['ECG', 'ecg', 'EKG', 'ekg', 'data', 'signal', 'val']
            
            for field in possible_fields:
                if field in mat_data and isinstance(mat_data[field], np.ndarray):
                    ecg_data = mat_data[field]
                    break
            
            if ecg_data is None:
                # If no standard field found, try the first array with appropriate shape
                for key, value in mat_data.items():
                    if isinstance(value, np.ndarray) and len(value.shape) >= 2:
                        if value.shape[0] > value.shape[1]:  # Assume time is the longer dimension
                            ecg_data = value
                        else:
                            ecg_data = value.T
                        break
        
        elif file_ext == '.csv':
            # For CSV files
            try:
                # Try with header
                df = pd.read_csv(file_path)
                
                # Check if there are actual headers or if first row is data
                first_row = df.iloc[0].values
                if all(isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).replace('-', '', 1).isdigit()) for x in first_row):
                    # First row seems to be data, reload without header
                    df = pd.read_csv(file_path, header=None)
            except:
                # Try without header
                df = pd.read_csv(file_path, header=None)
            
            # Convert to numpy array
            ecg_data = df.values
            
        elif file_ext == '.txt':
            # For text files, try various delimiters
            for delimiter in [',', '\t', ' ']:
                try:
                    # Load text file with the current delimiter
                    ecg_data = np.loadtxt(file_path, delimiter=delimiter)
                    break
                except:
                    continue
                    
            if 'ecg_data' not in locals():
                raise ValueError(f"Failed to parse text file {file_path} with any delimiter")
                
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Handle different possible layouts
        if len(ecg_data.shape) == 1:
            # Single lead ECG
            ecg_data = ecg_data.reshape(-1, 1)
        
        # Determine orientation - time should be the longer dimension
        if ecg_data.shape[0] < ecg_data.shape[1]:
            ecg_data = ecg_data.T
        
        # Ensure correct lead count
        if ecg_data.shape[1] < lead_count:
            # Pad with zeros if fewer leads than expected
            padding = np.zeros((ecg_data.shape[0], lead_count - ecg_data.shape[1]))
            ecg_data = np.hstack((ecg_data, padding))
        elif ecg_data.shape[1] > lead_count:
            # Use only the first lead_count leads
            ecg_data = ecg_data[:, :lead_count]
        
        # Resample to the desired length
        if ecg_data.shape[0] != sample_length:
            resampled_data = np.zeros((sample_length, ecg_data.shape[1]))
            for i in range(ecg_data.shape[1]):
                resampled_data[:, i] = signal.resample(ecg_data[:, i], sample_length)
            ecg_data = resampled_data
        
        return ecg_data
    
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        return None

def download_from_url(url, destination=None):
    """Download data from URL (supports regular URLs and Google Drive sharing links)"""
    if destination is None:
        # Create temp file
        _, destination = tempfile.mkstemp(suffix='.zip')
    
    print(f"Downloading data from {url}...")
    
    # Check if it's a Google Drive URL
    if 'drive.google.com' in url:
        # Extract the file ID from the URL
        if '/file/d/' in url:
            file_id = url.split('/file/d/')[1].split('/')[0]
        elif 'id=' in url:
            file_id = url.split('id=')[1].split('&')[0]
        else:
            raise ValueError(f"Couldn't extract file ID from Google Drive URL: {url}")
        
        # Download using gdown
        gdown.download(id=file_id, output=destination, quiet=False)
    else:
        # Regular URL
        urllib.request.urlretrieve(url, destination)
    
    print(f"Download complete: {destination}")
    return destination

#----------------------- Custom Callbacks -----------------------#

class SaveEpochCallback(tf.keras.callbacks.Callback):
    """Custom callback to save model after each epoch to Google Drive"""
    
    def __init__(self, checkpoint_dir, save_frequency=1, max_to_keep=5):
        super(SaveEpochCallback, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.save_frequency = save_frequency
        self.max_to_keep = max_to_keep
        self.saved_checkpoints = []
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_frequency == 0:
            # Create checkpoint directory if it doesn't exist
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            # Save model
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 
                f'model_epoch_{epoch+1:03d}_val_loss_{logs["val_cvd_prediction_loss"]:.4f}.h5'
            )
            
            self.model.save_weights(checkpoint_path)
            print(f"\nSaved checkpoint at epoch {epoch+1} to {checkpoint_path}")
            
            # Add to list of saved checkpoints
            self.saved_checkpoints.append(checkpoint_path)
            
            # Remove older checkpoints if we have too many
            if len(self.saved_checkpoints) > self.max_to_keep:
                oldest_checkpoint = self.saved_checkpoints.pop(0)
                try:
                    os.remove(oldest_checkpoint)
                    print(f"Removed old checkpoint: {oldest_checkpoint}")
                except:
                    print(f"Failed to remove old checkpoint: {oldest_checkpoint}")

#----------------------- UI Implementation -----------------------#

def create_cardiovisionnet_ui():
    """Create UI widgets for CardioVisionNet configuration"""
    
    # Data source widgets
    url_input = widgets.Text(
        value='',
        placeholder='URL to zip file (http:// or Google Drive link)',
        description='Data URL:',
        disabled=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%')
    )
    
    upload_button = widgets.Button(
        description='Upload Zip File',
        disabled=False,
        button_style='',
        tooltip='Upload from your computer',
        icon='upload'
    )
    
    drive_path_input = widgets.Text(
        value='',
        placeholder='Path to zip file in Google Drive (e.g., MyDrive/data.zip)',
        description='Drive Path:',
        disabled=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%')
    )
    
    labels_path_input = widgets.Text(
        value='',
        placeholder='Path to labels file (CSV or Excel)',
        description='Labels Path:',
        disabled=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%')
    )
    
    # Data configuration widgets
    lead_count_input = widgets.IntSlider(
        value=12,
        min=1,
        max=16,
        step=1,
        description='Lead Count:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        style={'description_width': 'initial'}
    )
    
    sample_length_input = widgets.IntSlider(
        value=5000,
        min=1000,
        max=10000,
        step=500,
        description='Sample Length:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        style={'description_width': 'initial'}
    )
    
    # Train/test/validation split widgets
    test_size_input = widgets.FloatSlider(
        value=0.2,
        min=0.1,
        max=0.5,
        step=0.05,
        description='Test Size:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        style={'description_width': 'initial'}
    )
    
    val_size_input = widgets.FloatSlider(
        value=0.2,
        min=0.1,
        max=0.5,
        step=0.05,
        description='Validation Size:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        style={'description_width': 'initial'}
    )
    
    # Model configuration widgets
    epochs_input = widgets.IntSlider(
        value=50,
        min=10,
        max=200,
        step=10,
        description='Epochs:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        style={'description_width': 'initial'}
    )
    
    batch_size_input = widgets.IntSlider(
        value=32,
        min=8,
        max=256,
        step=8,
        description='Batch Size:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        style={'description_width': 'initial'}
    )
    
    learning_rate_input = widgets.FloatLogSlider(
        value=0.001,
        base=10,
        min=-4,
        max=-2,
        step=0.1,
        description='Learning Rate:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.5f',
        style={'description_width': 'initial'}
    )
    
    # Checkpoint directory in Google Drive
    checkpoint_dir_input = widgets.Text(
        value='MyDrive/CardioVisionNet/checkpoints',
        placeholder='Path in Google Drive to save checkpoints',
        description='Checkpoint Dir:',
        disabled=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%')
    )
    
    # Save frequency
    save_frequency_input = widgets.IntSlider(
        value=1,
        min=1,
        max=10,
        step=1,
        description='Save Every N Epochs:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        style={'description_width': 'initial'}
    )
    
    # Advanced model parameters
    advanced_toggle = widgets.ToggleButton(
        value=False,
        description='Show Advanced Options',
        disabled=False,
        button_style='',
        tooltip='Toggle advanced options visibility',
        icon='cog'
    )
    
    weight_decay_input = widgets.FloatLogSlider(
        value=0.0001,
        base=10,
        min=-5,
        max=-2,
        step=0.1,
        description='Weight Decay:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.5f',
        style={'description_width': 'initial'}
    )
    
    dropout_rate_input = widgets.FloatSlider(
        value=0.3,
        min=0.1,
        max=0.5,
        step=0.05,
        description='Dropout Rate:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        style={'description_width': 'initial'}
    )
    
    filters_base_input = widgets.IntSlider(
        value=64,
        min=16,
        max=128,
        step=16,
        description='Filters Base:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        style={'description_width': 'initial'}
    )
    
    use_self_supervision_input = widgets.Checkbox(
        value=True,
        description='Use Self-Supervised Learning',
        disabled=False,
        indent=False,
        style={'description_width': 'initial'}
    )
    
    # Run button
    run_button = widgets.Button(
        description='Run CardioVisionNet',
        disabled=False,
        button_style='success',
        tooltip='Start training',
        icon='play'
    )
    
    # Stop button (will be used to stop training)
    stop_button = widgets.Button(
        description='Stop Training',
        disabled=True,
        button_style='danger',
        tooltip='Stop training',
        icon='stop'
    )
    
    # Status output
    status_output = widgets.Output()
    
    # Create tabs for different input methods
    data_source_tabs = widgets.Tab()
    data_source_tabs.children = [
        widgets.VBox([url_input]),
        widgets.VBox([upload_button]),
        widgets.VBox([drive_path_input])
    ]
    data_source_tabs.set_title(0, 'URL')
    data_source_tabs.set_title(1, 'Upload')
    data_source_tabs.set_title(2, 'Google Drive')
    
    # Advanced options container
    advanced_options = widgets.VBox([
        weight_decay_input,
        dropout_rate_input,
        filters_base_input,
        use_self_supervision_input
    ])
    advanced_options.layout.display = 'none'
    
    def toggle_advanced_options(change):
        if change['new']:
            advanced_options.layout.display = 'flex'
            advanced_toggle.description = 'Hide Advanced Options'
        else:
            advanced_options.layout.display = 'none'
            advanced_toggle.description = 'Show Advanced Options'
    
    advanced_toggle.observe(toggle_advanced_options, names='value')
    
    # Organize widgets into sections
    data_section = widgets.VBox([
        widgets.HTML(value="<h3>Data Source</h3>"),
        data_source_tabs,
        labels_path_input,
        widgets.HBox([lead_count_input, sample_length_input])
    ])
    
    split_section = widgets.VBox([
        widgets.HTML(value="<h3>Data Split</h3>"),
        widgets.HBox([test_size_input, val_size_input])
    ])
    
    training_section = widgets.VBox([
        widgets.HTML(value="<h3>Training Parameters</h3>"),
        epochs_input,
        batch_size_input,
        learning_rate_input,
        advanced_toggle,
        advanced_options
    ])
    
    checkpoint_section = widgets.VBox([
        widgets.HTML(value="<h3>Checkpoints</h3>"),
        checkpoint_dir_input,
        save_frequency_input
    ])
    
    button_section = widgets.HBox([run_button, stop_button])
    
    # Main UI
    ui = widgets.VBox([
        data_section,
        split_section,
        training_section,
        checkpoint_section,
        button_section,
        status_output
    ])
    
    # Training state
    training_in_progress = False
    model = None
    
    # Upload handler
    def handle_upload(b):
        with status_output:
            clear_output()
            print("Please upload your zip file...")
            uploaded = files.upload()
            
            if uploaded:
                # Get the filename of the uploaded file
                filename = list(uploaded.keys())[0]
                print(f"Uploaded file: {filename}")
                
                # Set the filename in the UI
                url_input.value = filename
                
                # Switch to URL tab
                data_source_tabs.selected_index = 0
    
    upload_button.on_click(handle_upload)
    
    # Run handler
    def handle_run(b):
        nonlocal training_in_progress, model
        if training_in_progress:
            with status_output:
                print("Training already in progress. Please wait or stop it first.")
            return
        
        # Update UI state
        run_button.disabled = True
        stop_button.disabled = False
        training_in_progress = True
        
        with status_output:
            clear_output()
            try:
                # Get data source based on selected tab
                selected_tab = data_source_tabs.selected_index
                
                data_source = None
                if selected_tab == 0:  # URL
                    url = url_input.value
                    if not url:
                        raise ValueError("Please enter a URL")
                    
                    # If it's a local file (from upload), use it directly
                    if os.path.exists(url):
                        data_source = url
                    elif 'drive.google.com' in url:
                        # Download from Google Drive
                        data_source = download_from_url(url)
                    else:
                        # Download from regular URL
                        data_source = download_from_url(url)
                
                elif selected_tab == 1:  # Upload
                    print("Please use the 'Upload' button to upload a file first")
                    training_in_progress = False
                    run_button.disabled = False
                    stop_button.disabled = True
                    return
                
                elif selected_tab == 2:  # Google Drive
                    drive_path = drive_path_input.value
                    if not drive_path:
                        raise ValueError("Please enter a path in Google Drive")
                    
                    # Make sure it's a full path
                    if not drive_path.startswith('/'):
                        full_path = os.path.join('/content/drive', drive_path)
                    else:
                        full_path = drive_path
                    
                    if not os.path.exists(full_path):
                        raise ValueError(f"File not found: {full_path}")
                    
                    data_source = full_path
                
                # Get labels path
                labels_path = labels_path_input.value
                if labels_path and not os.path.exists(labels_path):
                    # Check if it's a relative path in Drive
                    drive_labels_path = os.path.join('/content/drive', labels_path)
                    if os.path.exists(drive_labels_path):
                        labels_path = drive_labels_path
                    else:
                        raise ValueError(f"Labels file not found: {labels_path}")
                
                # Get other parameters
                lead_count = lead_count_input.value
                sample_length = sample_length_input.value
                test_size = test_size_input.value
                val_size = val_size_input.value
                epochs = epochs_input.value
                batch_size = batch_size_input.value
                learning_rate = learning_rate_input.value
                weight_decay = weight_decay_input.value
                dropout_rate = dropout_rate_input.value
                filters_base = filters_base_input.value
                use_self_supervision = use_self_supervision_input.value
                
                # Get checkpoint directory
                checkpoint_dir = checkpoint_dir_input.value
                save_frequency = save_frequency_input.value
                
                # Make sure checkpoint_dir is a full path
                if not checkpoint_dir.startswith('/'):
                    checkpoint_dir = os.path.join('/content/drive', checkpoint_dir)
                
                # Create checkpoint directory if it doesn't exist
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Load data
                print(f"Loading data from {data_source}...")
                if data_source.endswith('.zip'):
                    X_data, y_labels = load_data_from_zip(
                        data_source, labels_path, lead_count, sample_length
                    )
                elif os.path.isdir(data_source):
                    X_data, y_labels = load_data_from_folders(
                        data_source, labels_path, lead_count, sample_length
                    )
                else:
                    raise ValueError(f"Unsupported data source: {data_source}")
                
                print(f"Loaded data: {X_data.shape}, Labels: {y_labels.shape}")
                
                # Split data into train/validation/test
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X_data, y_labels, test_size=test_size, random_state=42, 
                    stratify=y_labels if len(y_labels.shape) > 1 else None
                )
                
                # Adjust validation size for remaining data
                effective_val_size = val_size / (1 - test_size)
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=effective_val_size, random_state=42,
                    stratify=y_temp if len(y_temp.shape) > 1 else None
                )
                
                print(f"Data split - Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
                
                # Convert labels to one-hot if needed
                if len(y_train.shape) == 1:
                    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
                    
                    # Convert string labels to integers
                    label_encoder = LabelEncoder()
                    y_train_encoded = label_encoder.fit_transform(y_train)
                    y_val_encoded = label_encoder.transform(y_val)
                    y_test_encoded = label_encoder.transform(y_test)
                    
                    # Convert integers to one-hot
                    n_classes = len(label_encoder.classes_)
                    y_train = np.eye(n_classes)[y_train_encoded]
                    y_val = np.eye(n_classes)[y_val_encoded]
                    y_test = np.eye(n_classes)[y_test_encoded]
                    
                    print(f"Converted labels to one-hot encoding with {n_classes} classes")
                    print(f"Classes: {label_encoder.classes_}")
                
                # Calculate class weights
                class_weights = None
                if len(y_train.shape) > 1:  # One-hot encoded
                    n_samples = len(y_train)
                    n_classes = y_train.shape[1]
                    class_counts = np.sum(y_train, axis=0)
                    class_weights = {i: (n_samples / (n_classes * count)) for i, count in enumerate(class_counts)}
                    print("Class weights:", class_weights)
                
                # Create log directory for TensorBoard
                log_dir = os.path.join(checkpoint_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
                os.makedirs(log_dir, exist_ok=True)
                
                # Create model
                print("Creating CardioVisionNet model...")
                model = CardioVisionNet(
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                    num_classes=y_train.shape[1] if len(y_train.shape) > 1 else np.max(y_train) + 1,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    dropout_rate=dropout_rate,
                    filters_base=filters_base,
                    use_self_supervision=use_self_supervision,
                    use_attention=True,
                    backbone='efficientnet',
                    model_dir=checkpoint_dir
                )
                
                # Create callbacks
                callbacks = [
                    SaveEpochCallback(checkpoint_dir, save_frequency=save_frequency),
                    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='500,520'),
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_cvd_prediction_loss', patience=15, restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_cvd_prediction_loss', factor=0.5, patience=5, min_lr=1e-6
                    )
                ]
                
                # Train the model
                print("Starting training...")
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    class_weights=class_weights
                )
                
                # Training complete - evaluate on test set
                print("\nTraining complete! Evaluating on test set...")
                eval_results = model.evaluate_model(X_test, y_test, verbose=1)
                
                # Save final model
                final_model_path = os.path.join(checkpoint_dir, 'final_model.h5')
                model.save(final_model_path)
                print(f"Final model saved to {final_model_path}")
                
                # Plot training history
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 2, 1)
                plt.plot(history.history['cvd_prediction_loss'])
                plt.plot(history.history['val_cvd_prediction_loss'])
                plt.title('Model Loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'], loc='upper right')
                
                plt.subplot(1, 2, 2)
                plt.plot(history.history['cvd_prediction_accuracy'])
                plt.plot(history.history['val_cvd_prediction_accuracy'])
                plt.title('Model Accuracy')
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'], loc='lower right')
                
                plt.tight_layout()
                history_plot_path = os.path.join(checkpoint_dir, 'training_history.png')
                plt.savefig(history_plot_path)
                plt.show()
                
                print(f"Training history plot saved to {history_plot_path}")
                
                # Generate evaluation visualizations
                print("\nGenerating evaluation visualizations...")
                try:
                    # Try to generate plots with the model's plot_results method
                    figures = model.plot_results(eval_results, save_dir=checkpoint_dir)
                    
                    # Show confusion matrix
                    if 'confusion_matrix' in figures:
                        plt.figure(figures['confusion_matrix'].number)
                        plt.show()
                    
                    # Show ROC curves
                    if 'roc_curves' in figures:
                        plt.figure(figures['roc_curves'].number)
                        plt.show()
                except Exception as vis_error:
                    print(f"Warning: Could not generate all visualizations: {str(vis_error)}")
                    
                    # Create basic confusion matrix
                    if 'confusion_matrix' in eval_results:
                        import seaborn as sns
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(
                            eval_results['confusion_matrix'], 
                            annot=True, 
                            fmt='d', 
                            cmap='Blues'
                        )
                        plt.title('Confusion Matrix')
                        plt.ylabel('True Label')
                        plt.xlabel('Predicted Label')
                        plt.tight_layout()
                        plt.savefig(os.path.join(checkpoint_dir, 'confusion_matrix.png'))
                        plt.show()
                
                print(f"\nAll results saved to {checkpoint_dir}")
                print("You can access the saved model and checkpoints in your Google Drive.")
                
            except Exception as e:
                print(f"Error: {str(e)}")
                import traceback
                traceback.print_exc()
            
            finally:
                # Reset UI state
                training_in_progress = False
                run_button.disabled = False
                stop_button.disabled = True
    
    # Connect event handlers
    run_button.on_click(handle_run)
    stop_button.on_click(handle_stop)