# CardioVisionNet

CardioVisionNet is an advanced deep learning model for cardiovascular disease (CVD) prediction from ECG images. This innovative approach combines computer vision, multi-pathway feature extraction, and explainable AI to create a state-of-the-art model for research and clinical applications.

## 📋 Features

- **ECG Image Processing**: Works directly with ECG chart images (PNG, JPG, etc.)
- **Multi-pathway Architecture**: Extracts features from different aspects of the ECG images
- **Multiple CNN Backbones**: Supports EfficientNet, ResNet, and DenseNet architectures
- **Explainable AI**: Includes GradCAM and Integrated Gradients for model interpretation
- **Self-supervised Learning**: Optional self-supervised pretraining for better performance
- **Attention Mechanisms**: Physiologically-informed attention to focus on important ECG regions
- **Research Ready**: Detailed metrics and visualizations for academic publications
- **Signal to Image Conversion**: Can convert raw ECG signals to standardized images

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/CardioVisionNet.git
cd CardioVisionNet

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Converting ECG Signals to Images

If you have raw ECG signal data, you can convert it to images:

```bash
python CardioVisionNet.py --mode convert --data_dir ./your_signals_folder --output_dir ./output_images --convert_signals
```

### Training a Model

Train CardioVisionNet on your ECG images:

```bash
python CardioVisionNet.py --mode train \
    --data_dir ./your_images_folder \
    --labels_file ./labels.csv \
    --output_dir ./model_output \
    --backbone efficientnet \
    --num_classes 5 \
    --batch_size 32 \
    --epochs 50 \
    --use_attention \
    --use_self_supervision
```

### Testing a Trained Model

Evaluate a trained model on a test dataset:

```bash
python CardioVisionNet.py --mode test \
    --data_dir ./test_images_folder \
    --labels_file ./test_labels.csv \
    --output_dir ./test_results \
    --model_path ./model_output/final_model
```

### Generating Explanations

Generate visualizations to explain model predictions:

```bash
python CardioVisionNet.py --mode explain \
    --data_dir ./example_images \
    --output_dir ./explanations \
    --model_path ./model_output/final_model \
    --explain_method gradcam
```

## 📊 Preparing Your Data

### Image Data Format

- The model expects ECG images in standard formats (PNG, JPG, etc.)
- Images will be resized to the specified dimensions (default: 224x224)
- Can be organized in a folder structure with subfolders for classes

### Labels File Format

The labels file should be a CSV or Excel file with at least two columns:
- A column identifying the image file (e.g., 'filename', 'id')
- A column with the class label (e.g., 'diagnosis', 'condition')

Example:
```
filename,diagnosis
ecg_001.png,normal
ecg_002.png,mi
ecg_003.png,arrhythmia
```

## 🔍 Model Architecture

CardioVisionNet uses a multi-pathway architecture:

1. **Image Preprocessing Module**: Enhances ECG image features
2. **Multi-pathway Feature Extraction**:
   - CNN Backbone Pathway: Extracts global features using pretrained networks
   - Local Pattern Pathway: Focuses on ECG-specific patterns at multiple scales
   - Wavelet Transform Pathway: Analyzes frequency and scale components
3. **Cross-Attention Fusion**: Combines features from different pathways
4. **Transformer Encoder**: Captures dependencies between features
5. **Physiological Attention**: Focuses on clinically relevant regions
6. **Meta-learning Adaptation**: Patient-specific adaptation
7. **Output Heads**: Classification with uncertainty estimation

## 📝 Citation

If you use this code for your research, please cite our work:

```
@article{cardiovisionnet2023,
  title={CardioVisionNet: An Advanced Deep Learning Architecture for ECG Image-based Cardiovascular Disease Prediction},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2023}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

- Your Name - Initial work - [Your GitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- This research was supported by [Your Institution/Grant]
- Thanks to the cardiovascular research community for datasets and benchmarks 