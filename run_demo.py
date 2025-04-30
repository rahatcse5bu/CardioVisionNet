#!/usr/bin/env python
# CardioVisionNet Demo Script
# This script demonstrates basic usage of CardioVisionNet

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from CardioVisionNet import CardioVisionNet, convert_signal_to_image, generate_gradcam_heatmap, apply_heatmap_to_image

def create_sample_data(output_dir='sample_data'):
    """Create sample ECG signal data for demonstration"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create some synthetic ECG data (simplified)
    num_samples = 500
    t = np.linspace(0, 10, num_samples)
    
    # Create 3 different types of synthetic ECGs
    ecg_types = {
        'normal': {
            'num': 5,
            'p_amp': 0.25,
            'qrs_amp': 1.0,
            't_amp': 0.35,
            'noise': 0.05,
            'hr': 60
        },
        'tachycardia': {
            'num': 5,
            'p_amp': 0.15,
            'qrs_amp': 1.1,
            't_amp': 0.25,
            'noise': 0.07,
            'hr': 120
        },
        'bradycardia': {
            'num': 5,
            'p_amp': 0.35,
            'qrs_amp': 0.9,
            't_amp': 0.4,
            'noise': 0.06,
            'hr': 40
        }
    }
    
    all_signals = []
    all_labels = []
    
    for ecg_class, params in ecg_types.items():
        for i in range(params['num']):
            # Generate synthetic ECG
            heart_rate = params['hr'] + np.random.normal(0, 5)  # Add slight variability
            beats_per_10s = heart_rate / 6  # 10 seconds of data
            
            # Generate base ECG
            ecg = np.zeros(num_samples)
            beat_width = int(num_samples / beats_per_10s)
            
            # Place QRS complexes
            for beat in range(int(beats_per_10s)):
                center = int(beat * beat_width + beat_width/2)
                # P wave
                p_start = center - int(beat_width * 0.2)
                p_end = center - int(beat_width * 0.1)
                ecg[p_start:p_end] += params['p_amp'] * np.sin(np.linspace(0, np.pi, p_end-p_start))
                
                # QRS complex
                qrs_start = center - int(beat_width * 0.05)
                qrs_mid = center
                qrs_end = center + int(beat_width * 0.05)
                ecg[qrs_start:qrs_mid] -= params['qrs_amp'] * 0.3 * np.linspace(0, 1, qrs_mid-qrs_start)
                ecg[qrs_mid:qrs_end] += params['qrs_amp'] * np.linspace(0, 1, qrs_end-qrs_mid)
                
                # T wave
                t_start = center + int(beat_width * 0.1)
                t_end = center + int(beat_width * 0.3)
                if t_end < num_samples:
                    ecg[t_start:t_end] += params['t_amp'] * np.sin(np.linspace(0, np.pi, t_end-t_start))
            
            # Add some noise
            ecg += np.random.normal(0, params['noise'], num_samples)
            
            # Create 12-lead ECG by adding variations to different leads
            ecg_12lead = np.zeros((num_samples, 12))
            for lead in range(12):
                lead_variation = np.random.normal(1, 0.2)  # Lead amplitude variation
                lead_noise = np.random.normal(0, 0.02, num_samples)  # Lead-specific noise
                ecg_12lead[:, lead] = ecg * lead_variation + lead_noise
            
            # Save as CSV
            np.savetxt(
                os.path.join(output_dir, f"{ecg_class}_{i+1}.csv"),
                ecg_12lead,
                delimiter=','
            )
            
            all_signals.append(ecg_12lead)
            all_labels.append(ecg_class)
    
    # Create labels file
    with open(os.path.join(output_dir, 'labels.csv'), 'w') as f:
        f.write('filename,diagnosis\n')
        for i, label in enumerate(all_labels):
            f.write(f"{label}_{(i%params['num'])+1}.csv,{label}\n")
    
    return all_signals, all_labels

def generate_sample_ecg_images(signals, labels, output_dir='sample_images'):
    """Convert sample signals to ECG images"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert each signal to an image
    image_paths = []
    for i, (signal, label) in enumerate(zip(signals, labels)):
        # Signal needs to be in the right shape [1, samples, leads]
        signal_batch = np.expand_dims(signal, axis=0)
        
        # Convert signal to image
        paths = convert_signal_to_image(
            signal_batch, 
            output_dir=output_dir,
            filename_prefix=f"{label}_{i%5+1}_",
            figsize=(10, 8),
            dpi=100,
            leads_per_row=4
        )
        
        image_paths.extend(paths)
    
    return image_paths

def demo_model(image_paths, output_dir='demo_results'):
    """Demonstrate model training and prediction"""
    os.makedirs(output_dir, exist_ok=True)
    
    # For demo purposes, we'll just create a model and make a prediction
    # (not actually training since that would require more data)
    
    # Create model
    print("Creating CardioVisionNet model...")
    model = CardioVisionNet(
        input_shape=(224, 224, 3),  # Standard image input
        num_classes=3,  # Our demo has 3 classes
        learning_rate=0.001,
        backbone='efficientnet',  # Use EfficientNet as backbone
        use_attention=True,
        use_self_supervision=False  # Disable for demo to save time
    )
    
    # For demo, we'll just show model architecture
    model.model.summary()
    
    # Make a sample prediction on first image
    if image_paths:
        print("\nDemonstrating prediction and explainability...")
        import cv2
        
        # Load and preprocess an image
        img_path = image_paths[0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        
        # Make a dummy prediction (model is not trained, just for demo)
        img_batch = np.expand_dims(img, axis=0)
        
        # This will give random predictions since model is not trained
        # In a real scenario, you would train the model first
        outputs = model.model(img_batch)
        pred_class = np.argmax(outputs[0][0])
        
        # Generate GradCAM explanation
        # (just for demonstration, not meaningful with untrained model)
        try:
            last_conv_layer = None
            for layer in reversed(model.model.layers):
                if 'conv' in layer.name.lower():
                    last_conv_layer = layer.name
                    break
                    
            if last_conv_layer:
                heatmap = generate_gradcam_heatmap(
                    model.model, img_batch, last_conv_layer, pred_class
                )
                
                # Apply heatmap to image
                superimposed_img = apply_heatmap_to_image(img, heatmap)
                
                # Save and display
                output_path = os.path.join(output_dir, 'demo_explanation.png')
                cv2.imwrite(output_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
                
                # Display the results
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(img)
                plt.title('Original ECG Image')
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(superimposed_img)
                plt.title('GradCAM Explanation (Demo)')
                plt.axis('off')
                
                plt.savefig(os.path.join(output_dir, 'comparison.png'))
                plt.close()
                
                print(f"Saved explanation visualization to {output_dir}")
        except Exception as e:
            print(f"Could not generate explanation: {str(e)}")

def run_demo():
    """Run the complete demo"""
    print("CardioVisionNet Demo")
    print("===================")
    
    print("\nStep 1: Creating sample ECG signals...")
    signals, labels = create_sample_data()
    
    print("\nStep 2: Converting signals to ECG images...")
    image_paths = generate_sample_ecg_images(signals, labels)
    
    print("\nStep 3: Demonstrating CardioVisionNet model...")
    demo_model(image_paths)
    
    print("\nDemo complete! Check the output directories for results.")
    print("In a real application, you would train the model on a larger dataset")
    print("and use it for predictions on new ECG images.")

if __name__ == "__main__":
    run_demo() 