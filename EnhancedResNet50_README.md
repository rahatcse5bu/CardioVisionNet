# Enhanced ResNet50 for ECG Image Classification

This repository contains an enhanced version of the ResNet50 architecture specifically optimized for ECG image classification. The model includes several modifications designed to improve evaluation metrics such as accuracy, precision, recall, F1-score, and specificity when working with ECG image datasets.

## Key Enhancements

The enhanced ResNet50 architecture includes the following optimizations:

### 1. ECG-Specific Feature Extraction

- **Multi-scale Convolutions**: Custom `ECGSpecificBlock` that uses varied kernel sizes (3×3, 5×5, 9×9, 15×15) to capture ECG patterns of different sizes:
  - Small kernels (3×3) for P, Q, S waves and fine details
  - Medium kernels (5×5) for QRS complexes
  - Large kernels (9×9) for T waves and ST segments
  - Extra-large kernels (15×15) for overall ECG morphology

- **Feature Pyramid Integration**: Extracts features from different levels of the ResNet50 backbone and combines them to maintain both high-level semantic information and fine-grained details.

### 2. Attention Mechanisms

- **ECG Attention Module**: Dual-attention mechanism that focuses on both:
  - Channel attention: Emphasizes important feature channels
  - Spatial attention: Focuses on relevant regions in the ECG image

- **Feature Calibration Module**: Domain-specific module that learns to detect important ECG components (P waves, QRS complexes, T waves, ST segments) and calibrates feature importance based on pathological significance.

### 3. Advanced Training Strategies

- **Focal Loss**: Addresses class imbalance by focusing more on difficult-to-classify examples
- **AdamW Optimizer**: Combines the benefits of Adam optimization with proper weight decay regularization
- **Cosine Learning Rate Schedule**: Gradually reduces learning rate with warm-up phase for more stable convergence
- **ECG-Specific Data Augmentation**: Limited rotations, no flips, and carefully controlled contrast/brightness adjustments that preserve ECG morphology

### 4. Multi-path Feature Fusion

- Combines features from different stages of the network to capture both:
  - Low-level features: ECG waveform shapes and morphological details
  - High-level features: Abstract patterns indicative of cardiac conditions

## Performance Improvements

The enhanced ResNet50 architecture is designed to improve the following metrics for ECG image classification:

- **Accuracy**: Better overall classification performance
- **Precision**: Reduced false positives through feature calibration
- **Recall**: Improved detection of pathological conditions via attention mechanisms
- **F1-Score**: Better balance between precision and recall
- **Specificity**: Reduced false positives through ECG-specific feature extraction

## Usage

```python
from EnhancedResNet50 import create_enhanced_resnet50, get_training_config

# Create the model
model = create_enhanced_resnet50(
    input_shape=(224, 224, 3),  # Standard image input size
    num_classes=5,              # Number of ECG classification categories
    weights='imagenet'          # Use ImageNet pre-training
)

# Get recommended training configuration
training_config = get_training_config(model)

# Compile the model
model.compile(
    optimizer=training_config['optimizer'],
    loss=training_config['loss'],
    metrics=training_config['metrics']
)

# Train the model with your data
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50)
```

## Demo

Run the included demo script to see the model in action:

```
python enhanced_resnet50_demo.py
```

The demo script:
1. Creates an enhanced ResNet50 model
2. Trains it on sample data (replace with your actual ECG image dataset)
3. Evaluates performance metrics
4. Generates visualizations of the confusion matrix and training history
5. Saves the trained model

## Implementation Details

### ECG-Specific Architectural Components

1. **Feature Extraction**:
   - Uses ResNet50 as the backbone network
   - Adds ECG-specific convolutional blocks optimized for cardiac patterns
   - Implements multi-scale feature fusion for comprehensive pattern recognition

2. **Attention Mechanisms**:
   - Channel attention focuses on important feature types
   - Spatial attention emphasizes diagnostically important regions
   - Feature calibration guides the model to pay more attention to pathologically significant patterns

3. **Classification Head**:
   - Multi-stream feature aggregation
   - Regularized dense layers with batch normalization
   - Dropout layers to prevent overfitting

## Requirements

- TensorFlow ≥ 2.9.0
- TensorFlow Addons ≥ 0.17.0
- NumPy ≥ 1.22.0
- Matplotlib ≥ 3.5.0
- scikit-learn ≥ 1.0.0
- OpenCV Python ≥ 4.5.0
- seaborn ≥ 0.11.0 