# Create a custom callback to save model after each epoch
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

# Create a UI for CardioVisionNet
def create_cardiovisionnet_ui():
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
                # Mount Google Drive if needed
                if not os.path.exists('/content/drive'):
                    drive.mount('/content/drive')
                
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
                        data_source = download_from_drive_url(url)
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
                
                # CardioVisionNet for ECG-based CVD Prediction
# Created with Claude 3.7 Sonnet

# Install required packages
!pip install tensorflow tensorflow-addons pywt scipy pandas matplotlib ipywidgets gdown plotly seaborn

# Download the CardioVisionNet module if not already present
import os
if not os.path.exists('cardiovisionnet.py'):
    !wget https://raw.githubusercontent.com/yourusername/cardiovisionnet/main/cardiovisionnet.py

# Import modules
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import ipywidgets as widgets
from google.colab import drive, files
import zipfile
import tempfile
import shutil
import gdown
import pandas as pd
from datetime import datetime

# Mount Google Drive
drive.mount('/content/drive')

# Import CardioVisionNet
# If module doesn't exist yet, create it using the code provided separately
exec(open('cardiovisionnet.py').read())

# Create folder data loader helper code
import os
import numpy as np
import zipfile
import tempfile
import shutil
import glob
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample

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
    """
    Load ECG data from a zip file
    
    Args:
        zip_file: Path to zip file containing ECG data
        labels_file: Optional path to CSV/Excel file with labels
        lead_count: Number of ECG leads to expect
        sample_length: Length to resample all ECG signals to
        
    Returns:
        X_data: ECG data as numpy array (samples, time_points, leads)
        y_labels: Labels as numpy array
    """
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
    """
    Load labels from CSV or Excel file
    
    Args:
        labels_file: Path to CSV or Excel file with labels
        
    Returns:
        Dictionary mapping file IDs to labels
    """
    # Determine file type and load accordingly
    file_ext = os.path.splitext(labels_file)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(labels_file)
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(labels_file)
    else:
        raise ValueError(f"Unsupported labels file format: {file_ext}")
    
    # Check for required columns
    required_columns = ['file_id', 'label']
    
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
    """
    Load ECG data from various file formats
    
    Args:
        file_path: Path to ECG file
        lead_count: Number of ECG leads to expect
        sample_length: Length to resample all signals to
        
    Returns:
        ECG data as numpy array (time_points, leads)
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.mat':
        return load_mat_file(file_path, lead_count, sample_length)
    elif file_ext == '.csv':
        return load_csv_file(file_path, lead_count, sample_length)
    elif file_ext == '.txt':
        return load_txt_file(file_path, lead_count, sample_length)
    elif file_ext == '.edf':
        return load_edf_file(file_path, lead_count, sample_length)
    else:
        print(f"Unsupported file format: {file_ext}")
        return None

def load_mat_file(file_path, lead_count=12, sample_length=5000):
    """Load ECG data from MATLAB .mat file"""
    try:
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
        
        if ecg_data is None:
            print(f"Could not find ECG data in {file_path}")
            return None
        
        # Handle different possible layouts
        if len(ecg_data.shape) == 1:
            # Single lead ECG
            ecg_data = ecg_data.reshape(-1, 1)
        elif len(ecg_data.shape) > 2:
            # Multi-dimensional, flatten to 2D (time, leads)
            ecg_data = ecg_data.reshape(ecg_data.shape[0], -1)
        
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
                resampled_data[:, i] = resample(ecg_data[:, i], sample_length)
            ecg_data = resampled_data
        
        return ecg_data
        
    except Exception as e:
        print(f"Error loading MAT file {file_path}: {str(e)}")
        return None

def load_csv_file(file_path, lead_count=12, sample_length=5000):
    """Load ECG data from CSV file"""
    try:
        # Load CSV (may have header or not)
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
                resampled_data[:, i] = resample(ecg_data[:, i], sample_length)
            ecg_data = resampled_data
        
        return ecg_data
        
    except Exception as e:
        print(f"Error loading CSV file {file_path}: {str(e)}")
        return None

def load_txt_file(file_path, lead_count=12, sample_length=5000):
    """Load ECG data from text file"""
    try:
        # Try various delimiters
        for delimiter in [',', '\t', ' ']:
            try:
                # Load text file with the current delimiter
                data = np.loadtxt(file_path, delimiter=delimiter)
                
                # If successful, process the data
                if len(data.shape) == 1:
                    # Single lead ECG
                    data = data.reshape(-1, 1)
                
                # Determine orientation - time should be the longer dimension
                if data.shape[0] < data.shape[1]:
                    data = data.T
                
                # Ensure correct lead count
                if data.shape[1] < lead_count:
                    # Pad with zeros if fewer leads than expected
                    padding = np.zeros((data.shape[0], lead_count - data.shape[1]))
                    data = np.hstack((data, padding))
                elif data.shape[1] > lead_count:
                    # Use only the first lead_count leads
                    data = data[:, :lead_count]
                
                # Resample to the desired length
                if data.shape[0] != sample_length:
                    resampled_data = np.zeros((sample_length, data.shape[1]))
                    for i in range(data.shape[1]):
                        resampled_data[:, i] = resample(data[:, i], sample_length)
                    data = resampled_data
                
                return data
                
            except:
                # Try next delimiter
                continue
        
        # If all delimiters failed
        print(f"Failed to parse text file {file_path} with any delimiter")
        return None
        
    except Exception as e:
        print(f"Error loading text file {file_path}: {str(e)}")
        return None

def load_edf_file(file_path, lead_count=12, sample_length=5000):
    """Load ECG data from EDF (European Data Format) file"""
    try:
        # EDF files require the pyedflib package
        import pyedflib
        
        # Load EDF file
        with pyedflib.EdfReader(file_path) as f:
            n_channels = f.signals_in_file
            channel_count = min(n_channels, lead_count)
            
            # Initialize array for data
            data = np.zeros((f.getNSamples()[0], channel_count))
            
            # Read each channel
            for i in range(channel_count):
                data[:, i] = f.readSignal(i)
            
            # Pad with zeros if fewer leads than expected
            if channel_count < lead_count:
                padding = np.zeros((data.shape[0], lead_count - channel_count))
                data = np.hstack((data, padding))
            
            # Resample to the desired length
            if data.shape[0] != sample_length:
                resampled_data = np.zeros((sample_length, data.shape[1]))
                for i in range(data.shape[1]):
                    resampled_data[:, i] = resample(data[:, i], sample_length)
                data = resampled_data
            
            return data
            
    except ImportError:
        print("pyedflib is required to load EDF files. Install with: pip install pyedflib")
        return None
    except Exception as e:
        print(f"Error loading EDF file {file_path}: {str(e)}")
        return None

# Download from Google Drive URL
def download_from_drive_url(url, destination=None):
    """
    Download a file from Google Drive
    
    Args:
        url: Google Drive URL
        destination: Where to save the file
        
    Returns:
        Path to the downloaded file
    """
    if destination is None:
        # Create temp file
        _, destination = tempfile.mkstemp()
    
    print(f"Downloading from Google Drive URL: {url}")
    
    # Extract file ID from URL
    if '/file/d/' in url:
        file_id = url.split('/file/d/')[1].split('/')[0]
    elif 'id=' in url:
        file_id = url.split('id=')[1].split('&')[0]
    else:
        raise ValueError(f"Couldn't extract file ID from URL: {url}")
    
    # Use gdown to download
    gdown.download(id=file_id, output=destination, quiet=False)
    
    return destination