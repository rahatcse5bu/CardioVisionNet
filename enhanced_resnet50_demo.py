"""
Demo script for using the Enhanced ResNet50 model for ECG image classification
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from EnhancedResNet50 import create_enhanced_resnet50, get_training_config, get_data_augmentation

def main():
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Set up parameters
    input_shape = (224, 224, 3)  # Standard image input size
    num_classes = 5  # Number of ECG classification categories
    batch_size = 32
    epochs = 50
    
    # Create model
    print("Creating enhanced ResNet50 model...")
    model = create_enhanced_resnet50(
        input_shape=input_shape,
        num_classes=num_classes,
        weights='imagenet'  # Use ImageNet pre-training
    )
    
    # Display model summary
    model.summary()
    
    # Get data augmentation
    data_augmentation = get_data_augmentation()
    
    # Example of loading data (replace with your actual data loading logic)
    print("Loading ECG image data...")
    # Simulating data loading - replace this with your actual data loading code
    # X_train, y_train, X_val, y_val, X_test, y_test = load_ecg_data()
    
    # For demonstration, let's create dummy data
    # In a real scenario, load your actual ECG image data
    X_train = np.random.rand(100, *input_shape)
    y_train = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, 100), num_classes)
    X_val = np.random.rand(20, *input_shape)
    y_val = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, 20), num_classes)
    X_test = np.random.rand(30, *input_shape)
    y_test = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, 30), num_classes)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Get training configuration
    training_config = get_training_config(model)
    
    # Compile model
    model.compile(
        optimizer=training_config['optimizer'],
        loss=training_config['loss'],
        metrics=training_config['metrics']
    )
    
    # Create model checkpoint callback
    checkpoint_dir = "model_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "enhanced_resnet50_{epoch:02d}_{val_accuracy:.4f}.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )
    
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Reduce learning rate on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model
    print("Training enhanced ResNet50 model...")
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint_callback, early_stopping, reduce_lr]
    )
    
    # Evaluate model
    print("Evaluating model on test data...")
    test_results = model.evaluate(X_test, y_test, verbose=1)
    
    # Print test results
    print("\nTest Results:")
    for metric_name, value in zip(model.metrics_names, test_results):
        print(f"{metric_name}: {value:.4f}")
    
    # Make predictions on test data
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    class_names = [f"Class {i}" for i in range(num_classes)]
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    print("Results saved to confusion_matrix.png and training_history.png")
    
    # Save the model
    model.save('enhanced_resnet50_ecg_model.h5')
    print("Model saved to enhanced_resnet50_ecg_model.h5")

def load_ecg_data():
    """
    Load and preprocess ECG image data
    
    This is a placeholder function - implement your own data loading logic
    to load your actual ECG image dataset
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Your data loading code here
    # For example:
    # 1. Load images from directory
    # 2. Split into train/val/test
    # 3. Preprocess images
    # 4. Convert labels to one-hot encoding
    
    # This is just a placeholder implementation
    input_shape = (224, 224, 3)
    num_classes = 5
    
    # Create dummy data
    X_train = np.random.rand(100, *input_shape)
    y_train = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, 100), num_classes)
    X_val = np.random.rand(20, *input_shape)
    y_val = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, 20), num_classes)
    X_test = np.random.rand(30, *input_shape)
    y_test = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, 30), num_classes)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    main() 