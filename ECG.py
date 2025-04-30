# ECG-CVD Colab UI with Google Drive Integration - COMPLETE IMPLEMENTATION
# This notebook provides a UI for training an ECG Image-to-Signal CVD Detection model

import os
import sys
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import AUC, Precision, Recall
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from google.colab import drive
from google.colab import files
import ipywidgets as widgets
from IPython.display import display, clear_output
from tqdm.notebook import tqdm

# Make sure the AttentionModule is fixed to handle dimension mismatches
class AttentionModule(Layer):
    """
    Attention module for focusing on relevant parts of the image during signal reconstruction.
    """
    def __init__(self, filters):
        super(AttentionModule, self).__init__()
        self.filters = filters

    def build(self, input_shape):
        # Get the number of input channels
        input_channels = input_shape[-1]

        # Create a projection layer if input channels don't match target filters
        self.projection = None
        if input_channels != self.filters:
            self.projection = Conv2D(self.filters, (1, 1), padding='same')

        # Attention components
        self.query = Conv2D(self.filters, (1, 1), padding='same')
        self.key = Conv2D(self.filters, (1, 1), padding='same')
        self.value = Conv2D(self.filters, (1, 1), padding='same')
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)

    def call(self, inputs):
        # Project inputs to desired filter size if needed
        if self.projection is not None:
            x = self.projection(inputs)
        else:
            x = inputs

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Reshape for matrix multiplication
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]

        query_reshaped = tf.reshape(query, [batch_size, -1, self.filters])
        key_reshaped = tf.reshape(key, [batch_size, -1, self.filters])
        value_reshaped = tf.reshape(value, [batch_size, -1, self.filters])

        # Attention map
        attention = tf.matmul(query_reshaped, key_reshaped, transpose_b=True)
        attention = tf.nn.softmax(attention, axis=-1)

        # Apply attention to value
        context = tf.matmul(attention, value_reshaped)
        context = tf.reshape(context, [batch_size, height, width, self.filters])

        # Residual connection with learnable weight - using the projected tensor
        output = self.gamma * context + x

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.filters)


# Define the other custom layers used in the model
class WaveletTransformLayer(Layer):
    """Layer for wavelet transform feature extraction from ECG signals."""
    def __init__(self):
        super(WaveletTransformLayer, self).__init__()

    def build(self, input_shape):
        # Create wavelet filters of different scales
        self.filters = []
        scales = [2, 4, 8, 16]

        for scale in scales:
            # Mexican hat wavelet approximation
            kernel_size = scale * 10 + 1
            if kernel_size % 2 == 0:
                kernel_size += 1

            self.filters.append(
                self.add_weight(
                    name=f'wavelet_filter_{scale}',
                    shape=(kernel_size, 1, input_shape[-1], input_shape[-1]),
                    initializer=self._mexican_hat_initializer(scale, kernel_size),
                    trainable=True
                )
            )

    def _mexican_hat_initializer(self, scale, kernel_size):
        def initializer(shape, dtype=None):
            # Create Mexican hat wavelet
            x = np.linspace(-scale * 5, scale * 5, kernel_size)
            y = (1.0 - (x / scale)**2) * np.exp(-(x / scale)**2 / 2.0)

            # Normalize
            y = y / np.sqrt(np.sum(y**2))

            # Reshape for convolution
            y = y.reshape((kernel_size, 1, 1, 1))
            y = np.repeat(y, shape[-2], axis=2)
            y = np.repeat(y, shape[-1], axis=3)

            return tf.convert_to_tensor(y, dtype=tf.float32)

        return initializer

    def call(self, inputs):
        # Apply wavelets at different scales
        outputs = []

        for wavelet_filter in self.filters:
            output = tf.nn.conv2d(
                inputs,
                wavelet_filter,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            outputs.append(output)

        # Concatenate all wavelet outputs
        return tf.concat(outputs, axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3] * len(self.filters))


class QRSFeatureExtractor(Layer):
    """Specialized layer for extracting QRS complex features from ECG signals."""
    def __init__(self):
        super(QRSFeatureExtractor, self).__init__()

    def build(self, input_shape):
        # QRS detection filters - specialized 1D convolutional kernels
        self.qrs_detector = Conv1D(
            32, kernel_size=17, padding='same', activation='relu')
        self.feature_extractor = Conv1D(
            64, kernel_size=5, padding='same', activation='relu')

    def call(self, inputs):
        x = self.qrs_detector(inputs)
        x = self.feature_extractor(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 64)


class HRVFeatureExtractor(Layer):
    """Layer for extracting heart rate variability features from ECG signals."""
    def __init__(self):
        super(HRVFeatureExtractor, self).__init__()

    def build(self, input_shape):
        # RR interval detection and processing layers
        self.rr_detector = Conv1D(32, kernel_size=25, padding='same', activation='relu')
        self.spectral_analyzer = Conv1D(64, kernel_size=51, padding='same', activation='relu')

    def call(self, inputs):
        x = self.rr_detector(inputs)
        x = self.spectral_analyzer(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 64)


# Main ECG Image-to-Signal CVD Model
class ECGImageToSignalCVDModel:
    """
    End-to-end model for cardiovascular disease early detection from ECG images.
    This pipeline consists of:
    1. Image preprocessing module
    2. Image-to-signal translation network (modified U-Net)
    3. Signal enhancement and feature extraction
    4. CVD classification and risk stratification network
    """

    def __init__(self, img_height=512, img_width=512, signal_length=5000):
        """
        Initialize the model with configurable parameters.

        Parameters:
        -----------
        img_height : int
            Height of input ECG images
        img_width : int
            Width of input ECG images
        signal_length : int
            Length of the output ECG signal
        """
        # Validate parameters
        if signal_length < 100:
            raise ValueError(f"Signal length must be at least 100 (got {signal_length})")

        self.img_height = img_height
        self.img_width = img_width
        self.signal_length = signal_length

        # Log parameters
        print(f"Initializing ECG-CVD model with:")
        print(f"- Image dimensions: {img_height}x{img_width}")
        print(f"- Signal length: {signal_length}")

        # Build the complete pipeline
        self.image_to_signal_model = self._build_image_to_signal_model()
        self.signal_cvd_model = self._build_signal_cvd_model()
        self.combined_model = self._build_combined_model()

        print("Model built successfully!")

    def _build_image_to_signal_model(self):
        """
        Build the image-to-signal translation model using a modified U-Net architecture.
        This model extracts ECG signal traces from images.
        """
        # Input layer
        inputs = Input((self.img_height, self.img_width, 3))

        # Encoder
        # Increased filters for better feature extraction
        e1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        e1 = Conv2D(64, (3, 3), activation='relu', padding='same')(e1)
        e1 = BatchNormalization()(e1)
        p1 = MaxPooling2D((2, 2))(e1)

        e2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
        e2 = Conv2D(128, (3, 3), activation='relu', padding='same')(e2)
        e2 = BatchNormalization()(e2)
        p2 = MaxPooling2D((2, 2))(e2)

        e3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
        e3 = Conv2D(256, (3, 3), activation='relu', padding='same')(e3)
        e3 = BatchNormalization()(e3)
        p3 = MaxPooling2D((2, 2))(e3)

        e4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
        e4 = Conv2D(512, (3, 3), activation='relu', padding='same')(e4)
        e4 = BatchNormalization()(e4)
        p4 = MaxPooling2D((2, 2))(e4)

        # Bridge
        b = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
        b = Conv2D(1024, (3, 3), activation='relu', padding='same')(b)
        b = BatchNormalization()(b)
        b = Dropout(0.3)(b) # Prevent overfitting

        # Decoder with attention mechanisms for better accuracy
        d4 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(b)
        d4 = concatenate([d4, e4])
        d4 = AttentionModule(512)(d4)  # Fixed attention module
        d4 = Conv2D(512, (3, 3), activation='relu', padding='same')(d4)
        d4 = Conv2D(512, (3, 3), activation='relu', padding='same')(d4)
        d4 = BatchNormalization()(d4)

        d3 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(d4)
        d3 = concatenate([d3, e3])
        d3 = AttentionModule(256)(d3)
        d3 = Conv2D(256, (3, 3), activation='relu', padding='same')(d3)
        d3 = Conv2D(256, (3, 3), activation='relu', padding='same')(d3)
        d3 = BatchNormalization()(d3)

        d2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(d3)
        d2 = concatenate([d2, e2])
        d2 = AttentionModule(128)(d2)
        d2 = Conv2D(128, (3, 3), activation='relu', padding='same')(d2)
        d2 = Conv2D(128, (3, 3), activation='relu', padding='same')(d2)
        d2 = BatchNormalization()(d2)

        d1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(d2)
        d1 = concatenate([d1, e1])
        d1 = AttentionModule(64)(d1)
        d1 = Conv2D(64, (3, 3), activation='relu', padding='same')(d1)
        d1 = Conv2D(64, (3, 3), activation='relu', padding='same')(d1)
        d1 = BatchNormalization()(d1)

        # Signal extraction layer - collapse 2D image to 1D signal
        flatten = Conv2D(1, (1, 1), activation='linear')(d1)

        # Generate multi-lead ECG signal output (shape transformation)
        # This is a critical step: converting 2D image features to 1D signal
        reshaped = Reshape((self.img_height, self.img_width))(flatten)
        signal_extraction = Lambda(lambda x: tf.reduce_mean(x, axis=1))(reshaped)
        
        # Add a Dense layer to convert the image width to signal length
        signal_dense = Dense(self.signal_length)(signal_extraction)
        
        # Reshape to the required signal shape
        signal_output = Reshape((self.signal_length, 1))(signal_dense)
        
        # Add a final 1D convolution for signal smoothing
        signal_output = Conv1D(1, kernel_size=3, padding='same', activation='linear')(signal_output)

        # Create model
        model = Model(inputs, signal_output)
        model.compile(optimizer=Adam(learning_rate=1e-4),
                      loss='mean_squared_error')

        return model

    def _build_signal_cvd_model(self):
        """
        Build a model that classifies CVD conditions from the extracted ECG signal.
        This model detects early signs of CVD from signal patterns.
        """
        # Input layer for the signal
        signal_input = Input(shape=(self.signal_length, 1))

        # 1D CNN for feature extraction from signal
        # Using residual connections and dilated convolutions for better feature extraction
        x = Conv1D(64, kernel_size=5, padding='same', activation='relu')(signal_input)
        x = BatchNormalization()(x)

        # Residual block 1 with dilated convolutions
        residual = x
        x = Conv1D(64, kernel_size=5, padding='same', dilation_rate=2, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(64, kernel_size=5, padding='same', dilation_rate=4, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Add()([x, residual])

        # Add wavelet transform layer for multi-scale analysis
        # For simplicity in this version, we'll use a standard Conv1D instead
        # as the WaveletTransformLayer is more complex
        wavelet_features = Conv1D(128, kernel_size=9, padding='same', activation='relu')(x)
        x = concatenate([x, wavelet_features])

        # Residual block 2
        residual = Conv1D(128, kernel_size=1, padding='same')(x)
        x = Conv1D(128, kernel_size=7, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(128, kernel_size=7, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Add()([x, residual])

        # Domain-specific feature extraction (ECG specific features)
        # Simplified versions of the QRS and HRV extractors
        qrs_features = Conv1D(64, kernel_size=17, padding='same', activation='relu')(x)
        hrv_features = Conv1D(64, kernel_size=25, padding='same', activation='relu')(x)

        # Combined features
        x = concatenate([x, qrs_features, hrv_features])

        # Sequence modeling
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Bidirectional(LSTM(128, return_sequences=False))(x)

        # Classification head for multi-class CVD detection
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)

        # Multi-task outputs for different CVD conditions
        # Primary output: Early CVD risk (binary)
        cvd_risk = Dense(1, activation='sigmoid', name='cvd_risk')(x)

        # Additional outputs for specific conditions
        arrhythmia = Dense(1, activation='sigmoid', name='arrhythmia')(x)
        myocardial_ischemia = Dense(1, activation='sigmoid', name='myocardial_ischemia')(x)
        heart_failure = Dense(1, activation='sigmoid', name='heart_failure')(x)

        # Create model with multiple outputs
        model = Model(
            inputs=signal_input,
            outputs=[cvd_risk, arrhythmia, myocardial_ischemia, heart_failure]
        )

        # Compile with weighted losses for multi-task learning
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss={
                'cvd_risk': 'binary_crossentropy',
                'arrhythmia': 'binary_crossentropy',
                'myocardial_ischemia': 'binary_crossentropy',
                'heart_failure': 'binary_crossentropy'
            },
            loss_weights={
                'cvd_risk': 1.0,
                'arrhythmia': 0.7,
                'myocardial_ischemia': 0.7,
                'heart_failure': 0.7
            },
            metrics={
                'cvd_risk': [AUC(), Precision(), Recall()],
                'arrhythmia': [AUC()],
                'myocardial_ischemia': [AUC()],
                'heart_failure': [AUC()]
            }
        )

        return model

    def _build_combined_model(self):
        """
        Build the end-to-end combined model that goes from ECG image to CVD detection.
        This combined model allows for joint training of both components.
        """
        # Input layer
        img_input = Input((self.img_height, self.img_width, 3))

        # Get reconstructed signal using the image-to-signal model
        reconstructed_signal = self.image_to_signal_model(img_input)
        # Name the output to match metrics
        reconstructed_signal = Lambda(lambda x: x, name="reconstructed_signal")(reconstructed_signal)

        # Feed reconstructed signal to CVD detection model
        cvd_outputs = self.signal_cvd_model(reconstructed_signal)
        
        # Explicitly name each output to match metric keys
        cvd_risk = Lambda(lambda x: x, name="cvd_risk")(cvd_outputs[0])
        arrhythmia = Lambda(lambda x: x, name="arrhythmia")(cvd_outputs[1])
        myocardial_ischemia = Lambda(lambda x: x, name="myocardial_ischemia")(cvd_outputs[2])
        heart_failure = Lambda(lambda x: x, name="heart_failure")(cvd_outputs[3])

        # Create end-to-end model
        model = Model(
            inputs=img_input,
            outputs=[
                reconstructed_signal,  # Output 1: Reconstructed ECG signal
                cvd_risk,              # Output 2: CVD risk
                arrhythmia,            # Output 3: Arrhythmia detection
                myocardial_ischemia,   # Output 4: Myocardial ischemia
                heart_failure          # Output 5: Heart failure
            ]
        )

        # Compile with appropriate loss weights
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=[
                'mean_squared_error',  # Loss for signal reconstruction
                'binary_crossentropy',  # Loss for CVD risk
                'binary_crossentropy',  # Loss for arrhythmia
                'binary_crossentropy',  # Loss for myocardial ischemia
                'binary_crossentropy'   # Loss for heart failure
            ],
            loss_weights=[
                0.5,  # Weight for signal reconstruction
                1.0,  # Weight for CVD risk (primary task)
                0.3,  # Weight for arrhythmia
                0.3,  # Weight for myocardial ischemia
                0.3   # Weight for heart failure
            ],
            metrics={
                "reconstructed_signal": [],  # No metrics for signal reconstruction
                "cvd_risk": [AUC(), Precision(), Recall()],
                "arrhythmia": [AUC()],
                "myocardial_ischemia": [AUC()],
                "heart_failure": [AUC()]
            }
        )

        return model

    def train(self, X_img, y_signal, y_cvd, batch_size=16, epochs=100):
        """
        Train the combined model with image data and corresponding labels.

        Parameters:
        -----------
        X_img : numpy array
            ECG images, shape (n_samples, img_height, img_width, channels)
        y_signal : numpy array
            Ground truth ECG signals, shape (n_samples, signal_length, 1)
        y_cvd : dict or tuple
            Dictionary or tuple containing the CVD labels:
            - 'cvd_risk': binary labels for CVD risk
            - 'arrhythmia': binary labels for arrhythmia
            - 'myocardial_ischemia': binary labels for myocardial ischemia
            - 'heart_failure': binary labels for heart failure
        """
        # First, perform the train-test split on X_img
        X_train, X_val = train_test_split(X_img, test_size=0.2, random_state=42)
        
        # Prepare output format based on the combined model
        if isinstance(y_cvd, dict):
            # Ensure all arrays have the same first dimension as X_img
            for key in y_cvd:
                if y_cvd[key].shape[0] != X_img.shape[0]:
                    # Replicate or truncate to match dimension
                    if y_cvd[key].shape[0] < X_img.shape[0]:
                        # Repeat the values to match the size
                        repeats = int(np.ceil(X_img.shape[0] / y_cvd[key].shape[0]))
                        y_cvd[key] = np.tile(y_cvd[key], (repeats, 1))[:X_img.shape[0]]
                    else:
                        # Truncate
                        y_cvd[key] = y_cvd[key][:X_img.shape[0]]
            
            # Ensure y_signal has the same first dimension as X_img
            if y_signal.shape[0] != X_img.shape[0]:
                if y_signal.shape[0] < X_img.shape[0]:
                    # Create a new array with zeros and copy available data
                    new_y_signal = np.zeros((X_img.shape[0], self.signal_length, 1))
                    new_y_signal[:y_signal.shape[0]] = y_signal
                    y_signal = new_y_signal
                else:
                    # Truncate
                    y_signal = y_signal[:X_img.shape[0]]
            
            # Split each label array using the same indices as X_img split
            train_indices = range(len(X_train))
            val_indices = range(len(X_train), len(X_img))
            
            y_signal_train = y_signal[:len(X_train)]
            y_signal_val = y_signal[len(X_train):]
            
            y_cvd_train = {
                'cvd_risk': y_cvd['cvd_risk'][:len(X_train)],
                'arrhythmia': y_cvd['arrhythmia'][:len(X_train)],
                'myocardial_ischemia': y_cvd['myocardial_ischemia'][:len(X_train)],
                'heart_failure': y_cvd['heart_failure'][:len(X_train)]
            }
            
            y_cvd_val = {
                'cvd_risk': y_cvd['cvd_risk'][len(X_train):],
                'arrhythmia': y_cvd['arrhythmia'][len(X_train):],
                'myocardial_ischemia': y_cvd['myocardial_ischemia'][len(X_train):],
                'heart_failure': y_cvd['heart_failure'][len(X_train):]
            }
            
            y_train = [
                y_signal_train,
                y_cvd_train['cvd_risk'],
                y_cvd_train['arrhythmia'],
                y_cvd_train['myocardial_ischemia'],
                y_cvd_train['heart_failure']
            ]
            
            y_val = [
                y_signal_val,
                y_cvd_val['cvd_risk'],
                y_cvd_val['arrhythmia'],
                y_cvd_val['myocardial_ischemia'],
                y_cvd_val['heart_failure']
            ]
        else:
            # Assuming y_cvd is a tuple of (cvd_risk, arrhythmia, mi, hf)
            # First ensure all components have the right shape
            if y_signal.shape[0] != X_img.shape[0]:
                if y_signal.shape[0] < X_img.shape[0]:
                    new_y_signal = np.zeros((X_img.shape[0], self.signal_length, 1))
                    new_y_signal[:y_signal.shape[0]] = y_signal
                    y_signal = new_y_signal
                else:
                    y_signal = y_signal[:X_img.shape[0]]
            
            y_cvd_fixed = []
            for label in y_cvd:
                if len(label) != X_img.shape[0]:
                    if len(label) < X_img.shape[0]:
                        repeats = int(np.ceil(X_img.shape[0] / len(label)))
                        label = np.tile(label, (repeats, 1))[:X_img.shape[0]]
                    else:
                        label = label[:X_img.shape[0]]
                y_cvd_fixed.append(label)
            
            # Split using same indices
            y_signal_train = y_signal[:len(X_train)]
            y_signal_val = y_signal[len(X_train):]
            
            y_cvd_train = [label[:len(X_train)] for label in y_cvd_fixed]
            y_cvd_val = [label[len(X_train):] for label in y_cvd_fixed]
            
            y_train = [y_signal_train] + y_cvd_train
            y_val = [y_signal_val] + y_cvd_val

        # Callbacks for training
        callbacks = [
            ModelCheckpoint(
                'ecg_cvd_model_best.h5',
                monitor='val_cvd_risk_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_cvd_risk_auc',
                mode='max',
                patience=15,
                verbose=1
            )
        ]

        # Train the model
        history = self.combined_model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def predict(self, X_img):
        """
        Predict CVD risk and conditions from ECG images.

        Parameters:
        -----------
        X_img : numpy array
            ECG images, shape (n_samples, img_height, img_width, channels)

        Returns:
        --------
        dict
            Dictionary containing:
            - 'reconstructed_signal': Reconstructed ECG signals
            - 'cvd_risk': Predicted CVD risk scores
            - 'arrhythmia': Predicted arrhythmia scores
            - 'myocardial_ischemia': Predicted myocardial ischemia scores
            - 'heart_failure': Predicted heart failure scores
        """
        predictions = self.combined_model.predict(X_img)

        return {
            'reconstructed_signal': predictions[0],
            'cvd_risk': predictions[1],
            'arrhythmia': predictions[2],
            'myocardial_ischemia': predictions[3],
            'heart_failure': predictions[4]
        }

    def evaluate(self, X_test, y_signal_test, y_cvd_test):
        """
        Evaluate the model on test data.

        Parameters:
        -----------
        X_test : numpy array
            Test ECG images
        y_signal_test : numpy array
            Ground truth test ECG signals
        y_cvd_test : dict or tuple
            Test CVD labels

        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        # Format test data
        if isinstance(y_cvd_test, dict):
            y_test_combined = [
                y_signal_test,
                y_cvd_test['cvd_risk'],
                y_cvd_test['arrhythmia'],
                y_cvd_test['myocardial_ischemia'],
                y_cvd_test['heart_failure']
            ]
        else:
            y_test_combined = [y_signal_test] + list(y_cvd_test)

        # Evaluate model
        metrics = self.combined_model.evaluate(X_test, y_test_combined, verbose=1)

        # Get predictions for ROC analysis
        predictions = self.predict(X_test)

        # Calculate ROC curve and AUC for CVD risk
        if isinstance(y_cvd_test, dict):
            y_true_cvd = y_cvd_test['cvd_risk']
        else:
            y_true_cvd = y_cvd_test[0]  # Assuming first element is CVD risk

        fpr, tpr, thresholds = roc_curve(y_true_cvd, predictions['cvd_risk'])
        roc_auc = auc(fpr, tpr)

        # Return detailed evaluation results
        return {
            'metrics': dict(zip(self.combined_model.metrics_names, metrics)),
            'roc': {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': roc_auc
            }
        }

    def visualize_results(self, X_img, y_signal=None, y_cvd=None, n_samples=5):
        """
        Visualize model predictions and ground truth (if provided).

        Parameters:
        -----------
        X_img : numpy array
            ECG images
        y_signal : numpy array, optional
            Ground truth ECG signals
        y_cvd : dict or tuple, optional
            Ground truth CVD labels
        n_samples : int
            Number of samples to visualize
        """
        predictions = self.predict(X_img[:n_samples])

        for i in range(n_samples):
            fig = plt.figure(figsize=(15, 10))

            # Plot original image
            plt.subplot(3, 1, 1)
            plt.imshow(X_img[i])
            plt.title(f'Sample {i+1}: Original ECG Image')
            plt.axis('off')

            # Plot reconstructed signal
            plt.subplot(3, 1, 2)
            plt.plot(predictions['reconstructed_signal'][i])
            plt.title('Reconstructed ECG Signal')

            # Plot ground truth signal if provided
            if y_signal is not None:
                plt.plot(y_signal[i], 'r--', alpha=0.7)
                plt.legend(['Reconstructed', 'Ground Truth'])

            # Show predictions
            plt.subplot(3, 1, 3)
            risk_scores = [
                predictions['cvd_risk'][i][0],
                predictions['arrhythmia'][i][0],
                predictions['myocardial_ischemia'][i][0],
                predictions['heart_failure'][i][0]
            ]

            conditions = ['CVD Risk', 'Arrhythmia', 'Myocardial Ischemia', 'Heart Failure']
            plt.bar(conditions, risk_scores)
            plt.ylim(0, 1)
            plt.title('Predicted CVD Risk Scores')

            plt.tight_layout()
            plt.show()


# Dataset loader with UI components for Colab
class ECGDatasetUI:
    """
    UI-based dataset loader for ECG Image-to-Signal CVD model in Google Colab.
    Provides widgets for loading, exploring, and preprocessing datasets.
    """

    def __init__(self):
        """Initialize the UI components."""
        self.is_drive_mounted = False
        self.dataset_path = None
        self.class_names = None
        self.data_dict = None
        self.model = None
        self.model_params = {
            'img_height': 256,  # Default to lower resolution for memory efficiency
            'img_width': 256,
            'signal_length': 5000,
            'batch_size': 16,
            'epochs': 50,
            'learning_rate': 1e-4
        }

    def mount_drive_ui(self):
        """Display UI for mounting Google Drive."""
        mount_button = widgets.Button(
            description="Mount Google Drive",
            button_style='info',
            icon='cloud-upload'
        )

        output = widgets.Output()

        def on_mount_button_clicked(b):
            with output:
                clear_output()
                drive.mount('/content/drive')
                print("Google Drive mounted successfully!")
                self.is_drive_mounted = True

                # After mounting, show the dataset loader UI
                self.show_dataset_loader_ui()

        mount_button.on_click(on_mount_button_clicked)

        display(widgets.VBox([
            widgets.HTML("<h3>Step 1: Mount Google Drive</h3>"),
            widgets.HTML("<p>Click the button below to mount your Google Drive to access your dataset.</p>"),
            mount_button,
            output
        ]))

    def show_dataset_loader_ui(self):
        """Display UI for loading dataset from Google Drive."""
        # Drive path input
        drive_path = widgets.Text(
            value='/content/drive/MyDrive/',
            placeholder='Enter path to ZIP file in Google Drive',
            description='ZIP Path:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='80%')
        )

        # Dataset type selection
        dataset_type = widgets.RadioButtons(
            options=['Multi-class folders', 'Custom format'],
            description='Dataset Type:',
            style={'description_width': 'initial'}
        )

        # Load button
        load_button = widgets.Button(
            description="Load Dataset",
            button_style='primary',
            icon='folder-open'
        )

        output = widgets.Output()

        def on_load_button_clicked(b):
            with output:
                clear_output()

                # Get path from the input field
                zip_path = drive_path.value

                try:
                    print(f"Loading dataset from {zip_path}...")

                    # Check if file exists
                    if not os.path.exists(zip_path):
                        print(f"Error: File not found at {zip_path}")
                        return

                    # Extract directory name for the output folder
                    zip_name = os.path.basename(zip_path).split('.')[0]
                    extract_path = f'/content/{zip_name}'

                    # Create extraction directory if it doesn't exist
                    os.makedirs(extract_path, exist_ok=True)

                    # Extract the ZIP file
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        for member in tqdm(zip_ref.infolist(), desc="Extracting files"):
                            zip_ref.extract(member, extract_path)

                    print(f"Dataset extracted to {extract_path}")
                    self.dataset_path = extract_path

                    # Based on dataset type, load the data
                    if dataset_type.value == 'Multi-class folders':
                        # Show multi-class dataset explorer UI
                        self.explore_multiclass_dataset()
                    else:
                        # Show custom dataset explorer UI
                        self.explore_custom_dataset()

                except Exception as e:
                    print(f"Error loading dataset: {str(e)}")
                    import traceback
                    traceback.print_exc()

        load_button.on_click(on_load_button_clicked)

        display(widgets.VBox([
            widgets.HTML("<h3>Step 2: Load Dataset</h3>"),
            widgets.HTML("<p>Enter the path to your ZIP file in Google Drive and select the dataset type.</p>"),
            drive_path,
            dataset_type,
            load_button,
            output
        ]))

    def explore_multiclass_dataset(self):
        """Display UI for exploring and processing a multi-class dataset."""
        if not self.dataset_path:
            print("No dataset loaded. Please load a dataset first.")
            return

        # Get all directories in the dataset path (classes)
        class_dirs = [d for d in os.listdir(self.dataset_path)
                     if os.path.isdir(os.path.join(self.dataset_path, d))]

        if not class_dirs:
            print(f"No class directories found in {self.dataset_path}")
            return

        print(f"Found {len(class_dirs)} classes: {', '.join(class_dirs)}")
        self.class_names = class_dirs

        # Dataset parameters UI
        image_size = widgets.IntSlider(
            value=256,
            min=128,
            max=512,
            step=32,
            description='Image Size:',
            style={'description_width': 'initial'},
            continuous_update=False
        )

        batch_size = widgets.IntSlider(
            value=16,
            min=4,
            max=64,
            step=4,
            description='Batch Size:',
            style={'description_width': 'initial'},
            continuous_update=False
        )

        validation_split = widgets.FloatSlider(
            value=0.2,
            min=0.1,
            max=0.5,
            step=0.05,
            description='Validation Split:',
            style={'description_width': 'initial'},
            continuous_update=False
        )

        augmentation = widgets.Checkbox(
            value=True,
            description='Use Data Augmentation',
            style={'description_width': 'initial'}
        )

        # Process button
        process_button = widgets.Button(
            description="Process Dataset",
            button_style='success',
            icon='cogs'
        )

        output = widgets.Output()

        def on_process_button_clicked(b):
            with output:
                clear_output()

                # Update model parameters
                self.model_params['img_height'] = image_size.value
                self.model_params['img_width'] = image_size.value
                self.model_params['batch_size'] = batch_size.value

                try:
                    print(f"Processing dataset with {len(class_dirs)} classes...")
                    print(f"Image size: {image_size.value}x{image_size.value}")
                    print(f"Batch size: {batch_size.value}")
                    print(f"Validation split: {validation_split.value}")
                    print(f"Data augmentation: {'Enabled' if augmentation.value else 'Disabled'}")

                    # Create TensorFlow dataset
                    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                        self.dataset_path,
                        validation_split=validation_split.value,
                        subset="training",
                        seed=42,
                        image_size=(image_size.value, image_size.value),
                        batch_size=batch_size.value
                    )

                    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                        self.dataset_path,
                        validation_split=validation_split.value,
                        subset="validation",
                        seed=42,
                        image_size=(image_size.value, image_size.value),
                        batch_size=batch_size.value
                    )

                    # Apply data augmentation if enabled
                    if augmentation.value:
                        data_augmentation = tf.keras.Sequential([
                            tf.keras.layers.RandomRotation(0.1),
                            tf.keras.layers.RandomZoom(0.1),
                            tf.keras.layers.RandomTranslation(0.1, 0.1),
                            tf.keras.layers.RandomContrast(0.1),
                        ])

                        train_ds = train_ds.map(
                            lambda x, y: (data_augmentation(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE
                        )

                    # Apply performance optimizations
                    AUTOTUNE = tf.data.AUTOTUNE
                    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
                    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

                    # Store dataset information
                    self.data_dict = {
                        'train_ds': train_ds,
                        'val_ds': val_ds,
                        'class_names': self.class_names,
                        'num_classes': len(self.class_names)
                    }

                    print(f"\nDataset processed successfully!")
                    print(f"Training set: {len(train_ds)} batches")
                    print(f"Validation set: {len(val_ds)} batches")

                    # Visualize some samples
                    self.visualize_dataset_samples()

                    # Show model configuration UI
                    self.show_model_config_ui()

                except Exception as e:
                    print(f"Error processing dataset: {str(e)}")
                    import traceback
                    traceback.print_exc()

        process_button.on_click(on_process_button_clicked)

        display(widgets.VBox([
            widgets.HTML("<h3>Step 3: Configure Dataset Processing</h3>"),
            widgets.HTML(f"<p>Found {len(class_dirs)} classes in the dataset. Configure processing parameters below.</p>"),
            image_size,
            batch_size,
            validation_split,
            augmentation,
            process_button,
            output
        ]))

    def explore_custom_dataset(self):
        """Display UI for exploring and processing a custom format dataset."""
        if not self.dataset_path:
            print("No dataset loaded. Please load a dataset first.")
            return

        # UI for specifying subdirectories and files
        images_dir = widgets.Text(
            value='images',
            placeholder='Folder containing ECG images',
            description='Images Directory:',
            style={'description_width': 'initial'}
        )

        signals_dir = widgets.Text(
            value='signals',
            placeholder='Folder containing signal data (optional)',
            description='Signals Directory:',
            style={'description_width': 'initial'}
        )

        labels_file = widgets.Text(
            value='labels.csv',
            placeholder='CSV file with labels (optional)',
            description='Labels File:',
            style={'description_width': 'initial'}
        )

        has_signals = widgets.Checkbox(
            value=False,
            description='Dataset includes signal data',
            style={'description_width': 'initial'}
        )

        image_size = widgets.IntSlider(
            value=256,
            min=128,
            max=512,
            step=32,
            description='Image Size:',
            style={'description_width': 'initial'},
            continuous_update=False
        )

        # Process button
        process_button = widgets.Button(
            description="Process Custom Dataset",
            button_style='success',
            icon='cogs'
        )

        output = widgets.Output()

        def on_process_button_clicked(b):
            with output:
                clear_output()

                # Update model parameters
                self.model_params['img_height'] = image_size.value
                self.model_params['img_width'] = image_size.value

                try:
                    # Check if specified directories exist
                    img_dir_path = os.path.join(self.dataset_path, images_dir.value)
                    if not os.path.exists(img_dir_path):
                        print(f"Error: Images directory '{img_dir_path}' not found")
                        return

                    print(f"Processing custom dataset...")
                    print(f"Images directory: {img_dir_path}")

                    # Signal directory handling
                    signal_dir_path = None
                    if has_signals.value:
                        signal_dir_path = os.path.join(self.dataset_path, signals_dir.value)
                        if not os.path.exists(signal_dir_path):
                            print(f"Warning: Signals directory '{signal_dir_path}' not found")
                            signal_dir_path = None
                        else:
                            print(f"Signals directory: {signal_dir_path}")

                    # Labels file handling
                    labels_path = None
                    if labels_file.value:
                        labels_path = os.path.join(self.dataset_path, labels_file.value)
                        if not os.path.exists(labels_path):
                            print(f"Warning: Labels file '{labels_path}' not found")
                            labels_path = None
                        else:
                            print(f"Labels file: {labels_path}")

                    # List image files
                    img_files = [f for f in os.listdir(img_dir_path)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

                    if not img_files:
                        print(f"Error: No image files found in {img_dir_path}")
                        return

                    print(f"Found {len(img_files)} image files")

                    # Load a sample image to display
                    sample_img_path = os.path.join(img_dir_path, img_files[0])
                    sample_img = tf.keras.preprocessing.image.load_img(
                        sample_img_path,
                        target_size=(image_size.value, image_size.value)
                    )
                    plt.figure(figsize=(6, 6))
                    plt.imshow(sample_img)
                    plt.title(f"Sample Image: {img_files[0]}")
                    plt.axis('off')
                    plt.show()

                    # If we have signals, load a sample signal
                    if signal_dir_path:
                        # Get corresponding signal file (assuming same name with different extension)
                        sample_signal_name = os.path.splitext(img_files[0])[0] + '.csv'
                        sample_signal_path = os.path.join(signal_dir_path, sample_signal_name)

                        if os.path.exists(sample_signal_path):
                            try:
                                # Load and plot signal
                                signal_data = np.loadtxt(sample_signal_path, delimiter=',')
                                plt.figure(figsize=(10, 4))
                                plt.plot(signal_data)
                                plt.title(f"Sample Signal: {sample_signal_name}")
                                plt.grid(True)
                                plt.show()
                            except Exception as e:
                                print(f"Error loading sample signal: {str(e)}")

                    # Show model configuration UI
                    print("\nDataset exploration complete. You can now configure the model.")
                    self.show_model_config_ui(custom_dataset=True)

                except Exception as e:
                    print(f"Error processing custom dataset: {str(e)}")
                    import traceback
                    traceback.print_exc()

        process_button.on_click(on_process_button_clicked)

        display(widgets.VBox([
            widgets.HTML("<h3>Step 3: Configure Custom Dataset</h3>"),
            widgets.HTML("<p>Specify the structure of your custom dataset.</p>"),
            images_dir,
            has_signals,
            signals_dir,
            labels_file,
            image_size,
            process_button,
            output
        ]))

    def visualize_dataset_samples(self, num_samples=5):
        """Visualize samples from the dataset."""
        if not self.data_dict or 'train_ds' not in self.data_dict:
            print("No dataset loaded or processed.")
            return

        # Get class names
        class_names = self.data_dict['class_names']

        # Take a batch of data
        for images, labels in self.data_dict['train_ds'].take(1):
            plt.figure(figsize=(12, 8))
            for i in range(min(num_samples, len(images))):
                plt.subplot(1, num_samples, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")
            plt.tight_layout()
            plt.show()

    def show_model_config_ui(self, custom_dataset=False):
        """Display UI for configuring the model parameters."""
        # Model configuration parameters
        img_height = widgets.IntSlider(
            value=self.model_params['img_height'],
            min=128,
            max=512,
            step=32,
            description='Image Height:',
            style={'description_width': 'initial'},
            continuous_update=False
        )

        img_width = widgets.IntSlider(
            value=self.model_params['img_width'],
            min=128,
            max=512,
            step=32,
            description='Image Width:',
            style={'description_width': 'initial'},
            continuous_update=False
        )

        signal_length = widgets.IntSlider(
            value=self.model_params['signal_length'],
            min=1000,
            max=10000,
            step=500,
            description='Signal Length:',
            style={'description_width': 'initial'},
            continuous_update=False
        )

        batch_size = widgets.IntSlider(
            value=self.model_params['batch_size'],
            min=4,
            max=64,
            step=4,
            description='Batch Size:',
            style={'description_width': 'initial'},
            continuous_update=False
        )

        epochs = widgets.IntSlider(
            value=self.model_params['epochs'],
            min=10,
            max=200,
            step=10,
            description='Epochs:',
            style={'description_width': 'initial'},
            continuous_update=False
        )

        learning_rate = widgets.FloatLogSlider(
            value=self.model_params['learning_rate'],
            base=10,
            min=-5,  # 10^-5 = 0.00001
            max=-2,  # 10^-2 = 0.01
            step=0.2,
            description='Learning Rate:',
            style={'description_width': 'initial'},
            continuous_update=False
        )

        # Build model button
        build_button = widgets.Button(
            description="Build Model",
            button_style='warning',
            icon='wrench'
        )

        output = widgets.Output()

        def on_build_button_clicked(b):
            with output:
                clear_output()

                # Update model parameters
                self.model_params['img_height'] = img_height.value
                self.model_params['img_width'] = img_width.value
                self.model_params['signal_length'] = signal_length.value
                self.model_params['batch_size'] = batch_size.value
                self.model_params['epochs'] = epochs.value
                self.model_params['learning_rate'] = learning_rate.value

                try:
                    print("Building ECG Image-to-Signal CVD model with parameters:")
                    print(f"Image size: {img_height.value}x{img_width.value}")
                    print(f"Signal length: {signal_length.value}")
                    print(f"Batch size: {batch_size.value}")
                    print(f"Epochs: {epochs.value}")
                    print(f"Learning rate: {learning_rate.value:.6f}")

                    # Build the model
                    self.model = ECGImageToSignalCVDModel(
                        img_height=img_height.value,
                        img_width=img_width.value,
                        signal_length=signal_length.value
                    )

                    print("\nModel built successfully!")

                    # Show model summary
                    print("\nImage-to-Signal Model Summary:")
                    self.model.image_to_signal_model.summary()

                    print("\nSignal-CVD Model Summary:")
                    self.model.signal_cvd_model.summary()

                    print("\nCombined Model Summary:")
                    self.model.combined_model.summary()

                    # Show training UI
                    self.show_training_ui(custom_dataset)

                except Exception as e:
                    print(f"Error building model: {str(e)}")
                    import traceback
                    traceback.print_exc()

        build_button.on_click(on_build_button_clicked)

        display(widgets.VBox([
            widgets.HTML("<h3>Step 4: Configure Model</h3>"),
            widgets.HTML("<p>Set the parameters for the ECG Image-to-Signal CVD model.</p>"),
            img_height,
            img_width,
            signal_length,
            batch_size,
            epochs,
            learning_rate,
            build_button,
            output
        ]))

    def show_training_ui(self, custom_dataset=False):
        """Display UI for training the model."""
        if not self.model:
            print("No model built. Please build a model first.")
            return

        # For custom datasets, we need additional data loading options
        if custom_dataset:
            data_option = widgets.RadioButtons(
                options=['Generate synthetic data for testing', 'Upload ECG and signal data'],
                description='Data Option:',
                style={'description_width': 'initial'}
            )

            upload_button = widgets.Button(
                description="Upload Files",
                button_style='info',
                icon='upload'
            )

            upload_output = widgets.Output()

            def on_upload_button_clicked(b):
                with upload_output:
                    clear_output()
                    print("Please use the Google Colab file upload feature to upload your files.")
                    print("Then specify the paths to your data in the fields below.")

            upload_button.on_click(on_upload_button_clicked)

            # Paths for uploaded data
            ecg_images_path = widgets.Text(
                value='',
                placeholder='Path to uploaded ECG images (e.g., /content/ecg_images.npy)',
                description='ECG Images Path:',
                style={'description_width': 'initial'}
            )

            signals_path = widgets.Text(
                value='',
                placeholder='Path to uploaded signal data (e.g., /content/signals.npy)',
                description='Signals Path:',
                style={'description_width': 'initial'}
            )

            labels_path = widgets.Text(
                value='',
                placeholder='Path to uploaded labels (e.g., /content/labels.csv)',
                description='Labels Path:',
                style={'description_width': 'initial'}
            )

            data_widgets = widgets.VBox([
                data_option,
                widgets.HBox([upload_button, upload_output]),
                ecg_images_path,
                signals_path,
                labels_path
            ])
        else:
            data_widgets = widgets.HTML("<p>The model will be trained on the dataset you processed earlier.</p>")

        # Training options
        use_early_stopping = widgets.Checkbox(
            value=True,
            description='Use early stopping',
            style={'description_width': 'initial'}
        )

        use_checkpointing = widgets.Checkbox(
            value=True,
            description='Save best model checkpoint',
            style={'description_width': 'initial'}
        )

        # Train button
        train_button = widgets.Button(
            description="Start Training",
            button_style='danger',
            icon='play'
        )

        output = widgets.Output()

        def on_train_button_clicked(b):
            with output:
                clear_output()

                try:
                    if custom_dataset:
                        # Handle custom dataset based on selected option
                        if data_option.value == 'Generate synthetic data for testing':
                            # Generate synthetic data for model testing
                            print("Generating synthetic data for model testing...")

                            # Create random images
                            X_img = np.random.rand(
                                100,
                                self.model_params['img_height'],
                                self.model_params['img_width'],
                                3
                            )

                            # Create random signals
                            y_signal = np.random.rand(
                                100,
                                self.model_params['signal_length'],
                                1
                            )

                            # Create random labels
                            y_cvd = {
                                'cvd_risk': np.random.randint(0, 2, size=(100, 1)),
                                'arrhythmia': np.random.randint(0, 2, size=(100, 1)),
                                'myocardial_ischemia': np.random.randint(0, 2, size=(100, 1)),
                                'heart_failure': np.random.randint(0, 2, size=(100, 1))
                            }

                        else:
                            # Load data from the specified paths
                            if not ecg_images_path.value:
                                print("Error: ECG images path must be specified")
                                return

                            print(f"Loading data from specified paths...")

                            # Load ECG images
                            X_img = np.load(ecg_images_path.value)
                            print(f"Loaded ECG images with shape: {X_img.shape}")

                            # Load signals if path is provided
                            if signals_path.value:
                                y_signal = np.load(signals_path.value)
                                print(f"Loaded signals with shape: {y_signal.shape}")
                            else:
                                # Generate dummy signals
                                y_signal = np.zeros((len(X_img), self.model_params['signal_length'], 1))
                                print("Using zero-filled dummy signals")

                            # Load labels if path is provided
                            if labels_path.value:
                                labels_df = pd.read_csv(labels_path.value)
                                print(f"Loaded labels with shape: {labels_df.shape}")

                                # Convert labels to dictionary format
                                y_cvd = {}
                                if 'cvd_risk' in labels_df.columns:
                                    y_cvd['cvd_risk'] = labels_df['cvd_risk'].values.reshape(-1, 1)
                                else:
                                    y_cvd['cvd_risk'] = np.zeros((len(X_img), 1))

                                if 'arrhythmia' in labels_df.columns:
                                    y_cvd['arrhythmia'] = labels_df['arrhythmia'].values.reshape(-1, 1)
                                else:
                                    y_cvd['arrhythmia'] = np.zeros((len(X_img), 1))

                                if 'myocardial_ischemia' in labels_df.columns:
                                    y_cvd['myocardial_ischemia'] = labels_df['myocardial_ischemia'].values.reshape(-1, 1)
                                else:
                                    y_cvd['myocardial_ischemia'] = np.zeros((len(X_img), 1))

                                if 'heart_failure' in labels_df.columns:
                                    y_cvd['heart_failure'] = labels_df['heart_failure'].values.reshape(-1, 1)
                                else:
                                    y_cvd['heart_failure'] = np.zeros((len(X_img), 1))
                            else:
                                # Generate dummy labels
                                y_cvd = {
                                    'cvd_risk': np.zeros((len(X_img), 1)),
                                    'arrhythmia': np.zeros((len(X_img), 1)),
                                    'myocardial_ischemia': np.zeros((len(X_img), 1)),
                                    'heart_failure': np.zeros((len(X_img), 1))
                                }
                                print("Using zero-filled dummy labels")
                    else:
                        # Use processed dataset
                        if not self.data_dict or 'train_ds' not in self.data_dict:
                            print("Error: No processed dataset found")
                            return

                        # For demonstration, convert TF dataset to numpy arrays
                        # In a real implementation, you would use the dataset directly
                        print("Converting TensorFlow dataset to numpy arrays...")

                        # Get all images and labels from the training dataset
                        X_img = []
                        y_labels = []

                        for images, labels in self.data_dict['train_ds']:
                            X_img.append(images.numpy())
                            y_labels.append(labels.numpy())

                        X_img = np.concatenate(X_img, axis=0)
                        y_labels = np.concatenate(y_labels, axis=0)

                        print(f"Converted {len(X_img)} images with shape {X_img.shape}")

                        # Create dummy signals and CVD labels for demonstration
                        # In a real implementation, you would have actual signal data
                        y_signal = np.zeros((len(X_img), self.model_params['signal_length'], 1))

                        # Convert class labels to binary CVD labels for demonstration
                        # In a real implementation, you would have actual CVD labels
                        y_cvd = {
                            'cvd_risk': (y_labels > 0).astype(np.float32).reshape(-1, 1),
                            'arrhythmia': np.zeros((len(X_img), 1)),
                            'myocardial_ischemia': np.zeros((len(X_img), 1)),
                            'heart_failure': np.zeros((len(X_img), 1))
                        }

                        print("Created dummy signals and converted labels to CVD format")

                    # Prepare callbacks
                    callbacks = []

                    if use_early_stopping.value:
                        early_stopping = EarlyStopping(
                            monitor='val_loss',
                            patience=10,
                            restore_best_weights=True,
                            verbose=1
                        )
                        callbacks.append(early_stopping)

                    if use_checkpointing.value:
                        checkpoint = ModelCheckpoint(
                            'ecg_cvd_model_best.h5',
                            monitor='val_loss',
                            save_best_only=True,
                            verbose=1
                        )
                        callbacks.append(checkpoint)

                    # Start training
                    print("\nStarting model training...")
                    print(f"Training with {len(X_img)} samples")
                    print(f"Batch size: {self.model_params['batch_size']}")
                    print(f"Epochs: {self.model_params['epochs']}")

                    # Set custom learning rate
                    optimizer = Adam(learning_rate=self.model_params['learning_rate'])
                    self.model.combined_model.optimizer = optimizer

                    # Train the model
                    history = self.model.train(
                        X_img=X_img,
                        y_signal=y_signal,
                        y_cvd=y_cvd,
                        batch_size=self.model_params['batch_size'],
                        epochs=self.model_params['epochs']
                    )

                    # Plot training history
                    plt.figure(figsize=(12, 4))

                    plt.subplot(1, 2, 1)
                    plt.plot(history.history['loss'])
                    plt.plot(history.history['val_loss'])
                    plt.title('Model Loss')
                    plt.ylabel('Loss')
                    plt.xlabel('Epoch')
                    plt.legend(['Train', 'Validation'], loc='upper right')

                    # Plot AUC if available
                    if 'cvd_risk_auc' in history.history:
                        plt.subplot(1, 2, 2)
                        plt.plot(history.history['cvd_risk_auc'])
                        plt.plot(history.history['val_cvd_risk_auc'])
                        plt.title('CVD Risk AUC')
                        plt.ylabel('AUC')
                        plt.xlabel('Epoch')
                        plt.legend(['Train', 'Validation'], loc='lower right')

                    plt.tight_layout()
                    plt.show()

                    # Save the model
                    self.model.combined_model.save_weights('ecg_cvd_model_final.h5')
                    print("\nModel training complete and saved as 'ecg_cvd_model_final.h5'")

                    # Show a sample prediction
                    print("\nGenerating sample predictions...")
                    sample_indices = np.random.choice(len(X_img), size=3, replace=False)
                    sample_images = X_img[sample_indices]

                    # Get predictions
                    predictions = self.model.predict(sample_images)

                    # Visualize predictions
                    self.model.visualize_results(
                        X_img=sample_images,
                        y_signal=y_signal[sample_indices] if y_signal is not None else None,
                        y_cvd=None,
                        n_samples=3
                    )

                    # Show inference UI
                    self.show_inference_ui()

                except Exception as e:
                    print(f"Error during training: {str(e)}")
                    import traceback
                    traceback.print_exc()

        train_button.on_click(on_train_button_clicked)

        display(widgets.VBox([
            widgets.HTML("<h3>Step 5: Train Model</h3>"),
            widgets.HTML("<p>Configure training options and start the training process.</p>"),
            data_widgets,
            use_early_stopping,
            use_checkpointing,
            train_button,
            output
        ]))

    def show_inference_ui(self):
        """Display UI for model inference on new images."""
        if not self.model:
            print("No trained model available. Please train a model first.")
            return

        # Inference options
        inference_type = widgets.RadioButtons(
            options=['Upload new image', 'Use sample from dataset', 'Generate synthetic test image'],
            description='Inference Type:',
            style={'description_width': 'initial'}
        )

        # Upload button for new images
        upload_button = widgets.Button(
            description="Upload Image",
            button_style='info',
            icon='upload'
        )

        upload_output = widgets.Output()

        def on_upload_button_clicked(b):
            with upload_output:
                clear_output()
                print("Please use the Google Colab file upload feature to upload your ECG image.")
                print("Then specify the path to your image in the field below.")

        upload_button.on_click(on_upload_button_clicked)

        # Path for uploaded image
        image_path = widgets.Text(
            value='',
            placeholder='Path to uploaded ECG image (e.g., /content/ecg_image.jpg)',
            description='Image Path:',
            style={'description_width': 'initial'}
        )

        # Sample index for using dataset sample
        sample_index = widgets.IntSlider(
            value=0,
            min=0,
            max=99,
            step=1,
            description='Sample Index:',
            style={'description_width': 'initial'},
            disabled=True
        )

        # Update UI based on inference type selection
        def on_inference_type_change(change):
            if change['new'] == 'Upload new image':
                upload_button.disabled = False
                image_path.disabled = False
                sample_index.disabled = True
            elif change['new'] == 'Use sample from dataset':
                upload_button.disabled = True
                image_path.disabled = True
                sample_index.disabled = False
            else:  # Generate synthetic test image
                upload_button.disabled = True
                image_path.disabled = True
                sample_index.disabled = True

        inference_type.observe(on_inference_type_change, names='value')

        # Run inference button
        run_button = widgets.Button(
            description="Run Inference",
            button_style='success',
            icon='laptop'
        )

        output = widgets.Output()

        def on_run_button_clicked(b):
            with output:
                clear_output()

                try:
                    if inference_type.value == 'Upload new image':
                        if not image_path.value:
                            print("Error: Please specify the path to the uploaded image")
                            return

                        print(f"Running inference on uploaded image at {image_path.value}...")

                        # Load and preprocess image
                        img = tf.keras.preprocessing.image.load_img(
                            image_path.value,
                            target_size=(self.model_params['img_height'], self.model_params['img_width'])
                        )
                        img_array = tf.keras.preprocessing.image.img_to_array(img)
                        img_array = img_array / 255.0  # Normalize
                        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

                    elif inference_type.value == 'Use sample from dataset':
                        if not hasattr(self, 'data_dict') or not self.data_dict:
                            print("Error: No dataset available")
                            return

                        print(f"Running inference on dataset sample with index {sample_index.value}...")

                        # Get sample from dataset
                        for images, _ in self.data_dict['train_ds'].take(1):
                            if sample_index.value >= len(images):
                                print(f"Error: Sample index {sample_index.value} is out of range")
                                return

                            img_array = np.expand_dims(images[sample_index.value].numpy(), axis=0)
                    else:
                        print("Generating synthetic test image...")

                        # Generate a synthetic test image with grid lines and wave patterns
                        img_array = np.ones((1, self.model_params['img_height'], self.model_params['img_width'], 3)) * 0.95

                        # Add grid lines
                        for i in range(0, self.model_params['img_height'], 20):
                            img_array[0, i, :, :] = 0.8

                        for j in range(0, self.model_params['img_width'], 20):
                            img_array[0, :, j, :] = 0.8

                        # Add simulated ECG trace
                        x = np.linspace(0, 10*np.pi, self.model_params['img_width'])

                        # Generate a sine wave with some variations to look like ECG
                        y = 0.5 * np.sin(x) + 0.2 * np.sin(2*x) + 0.1 * np.sin(3*x)

                        # Add some peaks
                        for i in range(10):
                            peak_loc = np.random.randint(50, self.model_params['img_width']-50)
                            peak_height = np.random.uniform(0.5, 1.5)

                            # Generate a QRS-like peak
                            peak = peak_height * np.exp(-0.05 * np.square(np.arange(self.model_params['img_width']) - peak_loc))
                            y += peak

                        # Normalize to image height
                        y = (y - np.min(y)) / (np.max(y) - np.min(y)) * 0.8
                        y = y * self.model_params['img_height'] * 0.6 + self.model_params['img_height'] * 0.2

                        # Add the trace to the image
                        for j in range(self.model_params['img_width']-1):
                            start_y = int(y[j])
                            end_y = int(y[j+1])

                            # Draw line between points
                            min_y = min(start_y, end_y)
                            max_y = max(start_y, end_y)

                            for k in range(min_y, max_y+1):
                                if 0 <= k < self.model_params['img_height']:
                                    img_array[0, k, j, 0] = 0.0  # Red channel
                                    img_array[0, k, j, 1] = 0.0  # Green channel
                                    img_array[0, k, j, 2] = 0.0  # Blue channel

                    # Display the input image
                    plt.figure(figsize=(8, 8))
                    plt.imshow(img_array[0])
                    plt.title("Input ECG Image")
                    plt.axis('off')
                    plt.show()

                    # Run inference
                    print("Processing image through the model...")
                    predictions = self.model.predict(img_array)

                    # Display reconstructed signal
                    plt.figure(figsize=(12, 4))
                    plt.plot(predictions['reconstructed_signal'][0])
                    plt.title("Reconstructed ECG Signal")
                    plt.grid(True)
                    plt.show()

                    # Display risk scores
                    risk_scores = [
                        predictions['cvd_risk'][0][0],
                        predictions['arrhythmia'][0][0],
                        predictions['myocardial_ischemia'][0][0],
                        predictions['heart_failure'][0][0]
                    ]

                    conditions = ['CVD Risk', 'Arrhythmia', 'Myocardial Ischemia', 'Heart Failure']

                    plt.figure(figsize=(10, 6))
                    bars = plt.bar(conditions, risk_scores, color=['skyblue', 'lightgreen', 'salmon', 'wheat'])

                    # Add risk level indicators
                    for i, bar in enumerate(bars):
                        risk_level = 'Low'
                        if risk_scores[i] > 0.7:
                            risk_level = 'High'
                        elif risk_scores[i] > 0.3:
                            risk_level = 'Medium'

                        plt.text(
                            bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.05,
                            f'{risk_scores[i]:.2f}\n({risk_level})',
                            ha='center',
                            fontweight='bold'
                        )

                    plt.ylim(0, 1.2)
                    plt.title('Predicted CVD Risk Scores')
                    plt.ylabel('Risk Score (0-1)')
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.show()

                    # Detailed analysis
                    print("\nDetailed Analysis:")
                    print("-----------------")

                    for i, condition in enumerate(conditions):
                        risk = risk_scores[i]
                        if risk > 0.7:
                            severity = "HIGH RISK"
                        elif risk > 0.3:
                            severity = "MODERATE RISK"
                        else:
                            severity = "LOW RISK"

                        print(f"{condition}: {risk:.2f} - {severity}")

                    print("\nSignal Characteristics:")
                    signal = predictions['reconstructed_signal'][0]

                    # Basic signal statistics
                    print(f"- Signal mean: {np.mean(signal):.4f}")
                    print(f"- Signal std: {np.std(signal):.4f}")
                    print(f"- Signal max: {np.max(signal):.4f}")
                    print(f"- Signal min: {np.min(signal):.4f}")

                    # Detect peaks (basic implementation)
                    from scipy.signal import find_peaks
                    peaks, _ = find_peaks(signal.flatten(), height=0.1, distance=50)

                    if len(peaks) > 0:
                        print(f"- Detected {len(peaks)} major peaks in the signal")
                        print(f"- Estimated heart rate: {len(peaks) * 60 / (len(signal) / 250):.1f} BPM (assuming 250 Hz)")

                    print("\nNote: This is a demonstration model. For accurate medical diagnosis, please consult a healthcare professional.")

                except Exception as e:
                    print(f"Error during inference: {str(e)}")
                    import traceback
                    traceback.print_exc()

        run_button.on_click(on_run_button_clicked)

        display(widgets.VBox([
            widgets.HTML("<h3>Step 6: Run Inference</h3>"),
            widgets.HTML("<p>Use the trained model to analyze ECG images and detect early signs of CVD.</p>"),
            inference_type,
            widgets.HBox([upload_button, upload_output]),
            image_path,
            sample_index,
            run_button,
            output
        ]))


# Main execution function to start the UI
def main():
    # Print header
    print("=" * 80)
    print("ECG Image-to-Signal CVD Detection Model with Google Drive Integration")
    print("=" * 80)
    print("This notebook provides a UI for training and using a deep learning model")
    print("that detects early signs of cardiovascular disease from ECG images.")
    print("\nThe model works in two stages:")
    print("1. Translates ECG images to digital signals using a modified U-Net")
    print("2. Analyzes the signals for early signs of CVD using domain-specific features")
    print("\nFollow the steps in the UI to:")
    print("- Mount Google Drive and load your dataset")
    print("- Configure and build the model")
    print("- Train the model on your data")
    print("- Run inference on new ECG images")
    print("=" * 80)

    # Initialize and start the UI
    ui = ECGDatasetUI()
    ui.mount_drive_ui()

if __name__ == "__main__":
    main()

# Make sure to execute this cell to display the UI
# If you're running in Google Colab, uncomment and run this:
# main()
