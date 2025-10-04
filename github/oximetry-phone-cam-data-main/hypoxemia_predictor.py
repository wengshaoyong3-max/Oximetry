import numpy as np
import pandas as pd
from pathlib import Path
import h5py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

FPS = 30
GT_INDEX = 3  # Use the 4th groundtruth channel (usually corresponds to SpO2)
# Use relative path, relative to script directory
H5_PATH = Path("../data/preprocessed/all_uw_data.h5")

def extract_features(signals):
    """
    Extract features from input signal windows.
    
    Parameters:
    signals (np.array): An array with shape (n_samples, n_timesteps, n_channels),
                        representing multiple signal samples. n_channels should be 3 (R, G, B).

    Returns:
    pd.DataFrame: DataFrame containing extracted features for each sample.
    """
    features_list = []
    # Iterate through each sample (e.g., every 3 seconds of video data)
    for i in range(signals.shape[0]):
        sample_features = {}
        # Process R, G, B channels separately
        for j, color in enumerate(['R', 'G', 'B']):
            channel_signal = signals[i, :, j]
            
            # Calculate DC component (mean) and AC component (standard deviation)
            dc_component = np.mean(channel_signal)
            ac_component = np.std(channel_signal)
            
            # Prevent division by zero
            if dc_component == 0:
                ac_dc_ratio = 0
            else:
                ac_dc_ratio = ac_component / dc_component

            sample_features[f'{color}_dc'] = dc_component
            sample_features[f'{color}_ac'] = ac_component
            sample_features[f'{color}_ac_dc_ratio'] = ac_dc_ratio
        
        # Calculate "Ratio of Ratios"
        # This is the core idea of traditional pulse oximetry
        if sample_features['B_ac_dc_ratio'] != 0:
            ror_rb = sample_features['R_ac_dc_ratio'] / sample_features['B_ac_dc_ratio']
        else:
            ror_rb = 0
            
        if sample_features['G_ac_dc_ratio'] != 0:
            ror_rg = sample_features['R_ac_dc_ratio'] / sample_features['G_ac_dc_ratio']
        else:
            ror_rg = 0

        sample_features['ror_rb'] = ror_rb
        sample_features['ror_rg'] = ror_rg
        
        features_list.append(sample_features)
        
    return pd.DataFrame(features_list)


def first_zero_index(arr1d: np.ndarray):
    idx = np.where(arr1d == 0)[0]
    return int(idx[0]) if idx.size > 0 else None


def trim_trailing_zeros(frames_3xT: np.ndarray, gt_1xS: np.ndarray):
    """Trim based on first zero (data by frames, GT by seconds), and ensure alignment: frames = seconds * FPS"""
    T_frames = frames_3xT.shape[1]
    S_secs = gt_1xS.shape[0]
    zf = first_zero_index(frames_3xT[0])  # Use red channel for determination
    zs = first_zero_index(gt_1xS)
    if zf is not None:
        T_frames = min(T_frames, zf)
    if zs is not None:
        S_secs = min(S_secs, zs)
    T_frames = min(T_frames, S_secs * FPS)
    S_secs = min(S_secs, T_frames // FPS)
    return frames_3xT[:, :T_frames], gt_1xS[:S_secs]


def load_h5(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"H5 file not found: {path}")
    with h5py.File(path, "r") as f:
        data = f["dataset"][:]        # (N, 6, T_frames)
        gt = f["groundtruth"][:]      # (N, 5, T_secs)
        # feat_names = list(f["dataset"].attrs.get("features_key", []))
        # gt_names = list(f["groundtruth"].attrs.get("gt_keys", []))
    return data, gt


def build_windows_from_h5(data: np.ndarray, gt: np.ndarray, win_frames: int = 90, hop_frames: int = 90,
                          by_hand: bool = True):
    """
    Build sliding window samples from H5 data (N,6,T) and (N,5,S).
    Returns: X_signals (n_samples, win_frames, 3), y_spo2 (n_samples,)
    """
    X_windows = []
    y_list = []
    N = data.shape[0]
    for pid in range(N):
        hands = [("left", [0, 1, 2]), ("right", [3, 4, 5])] if by_hand else [("both", [0, 1, 2, 3, 4, 5])]
        for hand_name, idxs in hands:
            if hand_name == "both":
                # If "both hands" needed, can be extended; here default to separate samples
                continue
            frames = data[pid, idxs, :]  # (3, T)
            frames, gt_secs = trim_trailing_zeros(frames, gt[pid, GT_INDEX, :])
            T = frames.shape[1]
            # Ensure alignment with second-level GT
            T = min(T, gt_secs.shape[0] * FPS)
            frames = frames[:, :T]
            gt_secs = gt_secs[: T // FPS]

            # Sliding window (3 seconds = 90 frames)
            i = 0
            while i + win_frames <= T:
                j = i + win_frames
                sec_from = i // FPS
                sec_to = j // FPS
                seg = frames[:, i:j].T  # (win, 3)
                X_windows.append(seg)
                y_list.append(float(np.mean(gt_secs[sec_from:sec_to])))
                i += hop_frames

    if not X_windows:
        raise RuntimeError("No samples generated, please check H5 content or window parameters (win/hop).")
    X_signals = np.stack(X_windows, axis=0)  # (n_samples, win, 3)
    y_spo2 = np.array(y_list, dtype=np.float64)
    return X_signals, y_spo2

# --- 1. Load from H5 and build samples ---
# Read: data/preprocessed/all_uw_data.h5
# Build: Extract R/G/B average pixel time series by 3-second window (90 frames) sliding window,
# and use the average value of ground truth SpO2 for the same time period as label.
print(f"Using H5 path: {H5_PATH}")
data, gt = load_h5(H5_PATH)
X_raw, y = build_windows_from_h5(data, gt, win_frames=90, hop_frames=90, by_hand=True)
print("Raw signal data shape:", X_raw.shape)  # (n_samples, 90, 3)
print("True SpO2 label shape:", y.shape)
print("-" * 30)

# --- 2. Feature extraction ---
X_features = extract_features(X_raw)
print("Feature data shape after extraction:", X_features.shape)
print("Extracted features example:")
print(X_features.head())
print("-" * 30)


# --- 3. Split training and test sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42
)

print(f"Training set samples: {len(X_train)}")
print(f"Test set samples: {len(X_test)}")
print("-" * 30)

# --- 4. Train linear regression model ---
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training completed.")
print("Model coefficients (weights):", model.coef_)
print("Model intercept (bias):", model.intercept_)
print("-" * 30)


# --- 5. Evaluate model ---
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model evaluation results on test set:")
print(f"Mean Absolute Error (MAE): {mae:.2f}% SpO2")
print(f"R-squared (R2 Score): {r2:.2f}")
print("-" * 30)


# --- 7. Hypoxemia prediction function ---
def predict_hypoxemia(input_file, trained_model):
    """
    Receive a file and perform hypoxemia prediction.

    Parameters:
    input_file (File object): User uploaded file object.
    trained_model (LinearRegression): Trained linear regression model.

    Returns:
    str: Predicted SpO2 value.
    str: Classification result ("Normal" or "Potential Hypoxemia").
    """
    # 1. Load and process input data
    # Assume input is CSV file containing R, G, B three-column signal data
    try:
        df = pd.read_csv(input_file.name)
        # Ensure R, G, B columns are included
        if not {'R', 'G', 'B'}.issubset(df.columns):
            return "Error", "CSV file must contain 'R', 'G', 'B' columns."
        
        # Convert data to Numpy array format required by model (1 sample, n frames, 3 channels)
        signal_data = df[['R', 'G', 'B']].to_numpy()
        
        # Trim or pad to 90 frames
        if len(signal_data) > 90:
            signal_data = signal_data[:90]
        elif len(signal_data) < 90:
            padding = np.zeros((90 - len(signal_data), 3))
            signal_data = np.vstack([signal_data, padding])
            
        signal_sample = np.expand_dims(signal_data, axis=0)
        
    except Exception as e:
        return "Error", f"File processing failed: {e}"

    # 2. Feature extraction
    features = extract_features(signal_sample)

    # 3. Use model for regression prediction
    predicted_spo2 = trained_model.predict(features)[0]

    # 4. Classification based on threshold
    # The paper uses 90% as a common hypoxemia threshold
    classification_threshold = 90.0
    
    if predicted_spo2 < classification_threshold:
        classification_result = "Warning: Potential Hypoxemia Detected"
    else:
        classification_result = "Normal"
        
    return f"{predicted_spo2:.2f} %", classification_result


# --- Improved prediction function ---
def predict_hypoxemia_improved(input_file, trained_model):
    """
    Improved hypoxemia prediction function with result saving functionality.
    """
    try:
        df = pd.read_csv(input_file)
        # Ensure R, G, B columns are included
        if not {'R', 'G', 'B'}.issubset(df.columns):
            return "Error", "CSV file must contain 'R', 'G', 'B' columns."
        
        # Convert data to Numpy array format required by model (1 sample, n frames, 3 channels)
        signal_data = df[['R', 'G', 'B']].to_numpy()
        
        # Trim or pad to 90 frames
        if len(signal_data) > 90:
            signal_data = signal_data[:90]
        elif len(signal_data) < 90:
            padding = np.zeros((90 - len(signal_data), 3))
            signal_data = np.vstack([signal_data, padding])
            
        signal_sample = np.expand_dims(signal_data, axis=0)
        
    except Exception as e:
        return "Error", f"File processing failed: {e}"

    # 2. Feature extraction
    features = extract_features(signal_sample)

    # 3. Use model for regression prediction
    predicted_spo2 = trained_model.predict(features)[0]

    # 4. Classification based on threshold
    classification_threshold = 90.0
    
    if predicted_spo2 < classification_threshold:
        classification_result = "Warning: Potential Hypoxemia Detected"
    else:
        classification_result = "Normal"
    
    # 5. Save prediction results to CSV file
    save_prediction_to_csv(predicted_spo2, classification_result)
        
    return f"{predicted_spo2:.2f} %", classification_result


def save_prediction_to_csv(predicted_spo2, classification_result):
    """
    Save prediction results to CSV file.
    """
    try:
        import datetime
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create new prediction record
        new_record = {
            'timestamp': current_time,
            'predicted_spo2': predicted_spo2,
            'classification': classification_result
        }
        
        # Save to CSV file
        output_path = Path("../data/preprocessed/linreg_preds.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists
        if output_path.exists():
            # Read existing data and append
            df_existing = pd.read_csv(output_path)
            df_new = pd.DataFrame([new_record])
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            # Create new file
            df_combined = pd.DataFrame([new_record])
        
        df_combined.to_csv(output_path, index=False)
        print(f"Prediction results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error saving prediction results: {e}")


# Create wrapper function for Gradio interface
def gradio_interface_function(input_file):
    """
    Wrapper function for Gradio interface.
    """
    if input_file is None:
        return "Please upload a CSV file", "No input file"
    
    return predict_hypoxemia_improved(input_file.name, model)


# --- 8. Launch Gradio Web Interface ---
try:
    import gradio as gr
    
    print("--- Launching Gradio Web Interface ---")
    
    # Create a simple description for the interface
    description = """
    This software is a learning project based on the paper "Smartphone camera oximetry in an induced hypoxemia study".
    It uses a **simplified linear regression model** to predict blood oxygen saturation (SpO2) and screen for hypoxemia based on uploaded RGB signal data.
    **Please note: This is a technical demonstration prototype and must not be used for actual medical diagnosis.**
    
    ### Usage Instructions:
    1. Prepare a CSV file containing R, G, B columns
    2. Upload the file for prediction
    3. View predicted SpO2 values and health status assessment
    4. Prediction results are automatically saved to data/preprocessed/linreg_preds.csv
    """
    
    # Create and launch Gradio interface
    iface = gr.Interface(
        fn=gradio_interface_function,
        inputs=gr.File(label="Upload RGB Signal CSV File", file_types=[".csv"]),
        outputs=[
            gr.Textbox(label="Predicted SpO2 Value"),
            gr.Textbox(label="Health Status Assessment")
        ],
        title="Hypoxemia Prediction Software Prototype",
        description=description,
        allow_flagging="never",
        theme=gr.themes.Soft()
    )
    
    # Launch interface
    print("Starting Web interface...")
    print("Interface will open automatically in browser, or manually access the displayed local URL")
    iface.launch(share=False, debug=True)
    
except ImportError:
    print("Gradio not installed. Please run the following command to install:")
    print("pip install gradio")
    print("Then rerun this script: python hypoxemia_predictor.py")
    
    # Provide command-line version prediction functionality
    print("\n--- Command Line Version ---")
    print("You can use the following function for prediction:")
    print("predict_hypoxemia_improved('your_file.csv', model)")
    
except Exception as e:
    print(f"Error launching Gradio interface: {e}")
    print("Using command line version for prediction")


# --- 6. Result visualization ---
plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred, line_kws={"color": "red", "lw": 2})
plt.xlabel("Ground Truth SpO2 (%)")
plt.ylabel("Predicted SpO2 (%)")
plt.title("Linear Regression: Prediction vs Ground Truth")
plt.grid(True)
plt.show()
