import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import sys
import os
import torchaudio
import torchaudio.transforms as T
from colorama import Fore, Style, init

# Initialize Colors
init()

# ==========================================
# CONFIGURATION
# ==========================================
# Paths
MODEL_PATH = "models/mn10_mydata_epoch_13.pt"
# Make sure this points to your specific results CSV
CSV_PATH = "outputs/all_samples_T8p49_th0p54_coswara_20260117_230855.csv"
OUTPUT_FILE = "models/mahalanobis_params.pkl"

# Audio Params (MUST MATCH LIVE_DEMO.PY)
MODEL_SAMPLE_RATE = 32000
N_FFT = 1024
HOP_LENGTH = 320
N_MELS = 128
WIN_LENGTH = 800
FMIN = 0.0
FMAX = 15000

# ==========================================
# PREPROCESSING ENGINE
# ==========================================
def preprocess(file_path):
    """
    Exact replica of the live_demo preprocessing pipeline.
    """
    try:
        waveform, sr = torchaudio.load(file_path)
    except Exception as e:
        print(f"{Fore.RED}Error loading {file_path}: {e}{Style.RESET_ALL}")
        return None

    # 1. Stereo to Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 2. Resample
    if sr != MODEL_SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=MODEL_SAMPLE_RATE)
        waveform = resampler(waveform)

    # 3. Peak Normalization (Matches inference logic)
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0.01:
        waveform = waveform / max_val

    # 4. Pre-emphasis Filter
    waveform = torch.nn.functional.conv1d(
        waveform.unsqueeze(0), 
        torch.tensor([[[-.97, 1]]])
    ).squeeze(0)

    # 5. Mel Spectrogram
    window = torch.hann_window(WIN_LENGTH, periodic=False)
    stft = torch.stft(
        waveform, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH, 
        win_length=WIN_LENGTH, 
        center=True, 
        normalized=False, 
        window=window, 
        return_complex=False
    )
    power_spec = (stft ** 2).sum(dim=-1)
    
    mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
        N_MELS, N_FFT, MODEL_SAMPLE_RATE, FMIN, FMAX, 
        vtln_low=100.0, vtln_high=-500., vtln_warp_factor=1.0
    )
    mel_basis = torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0)
    melspec = torch.matmul(mel_basis, power_spec)
    
    # 6. Log & Normalize
    melspec = (melspec + 0.00001).log()
    melspec = (melspec + 4.5) / 5.0

    return melspec.unsqueeze(0)

# ==========================================
# MODEL LOADER
# ==========================================
sys.path.append("models/mn") # Ensure this points to where model.py lives
from model import get_model 

def get_feature_extractor():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = get_model(num_classes=2, width_mult=1.0, head_type="mlp", se_dims='c').to(device)
    
    # Load Weights
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    model.eval()
    
    return model, device

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load Data
    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # 2. Filter High-Quality True Positives
    # Note: Adjust column names if your CSV is different
    if 'cough_prob_calibrated' not in df.columns:
        print(f"{Fore.YELLOW}Warning: 'cough_prob_calibrated' not found. Using 'cough_probability'.{Style.RESET_ALL}")
        prob_col = 'cough_probability'
    else:
        prob_col = 'cough_prob_calibrated'

    tp_df = df[(df['result'] == 'TP') & (df[prob_col] > 0.80)]
    print(f"Found {len(tp_df)} High-Confidence True Positives.")

    if len(tp_df) < 50:
        print(f"{Fore.RED}Error: Not enough samples ({len(tp_df)}) to build a robust matrix. Need at least 50.{Style.RESET_ALL}")
        # sys.exit(1) # Uncomment to enforce strictness

    # 3. Extract Features
    model, device = get_feature_extractor()
    features_list = []
    
    count = 0
    for idx, row in tp_df.iterrows():
        fpath = row['file_path']
        
        if not os.path.exists(fpath):
            # Try relative path if absolute fails
            fpath = os.path.basename(fpath) 
            if not os.path.exists(fpath):
                continue

        # Preprocess
        input_tensor = preprocess(fpath)
        if input_tensor is None: continue
        
        with torch.no_grad():
            logits, features = model(input_tensor.to(device))
            features_list.append(features.cpu().numpy().flatten())
        
        count += 1
        print(f"Processed {count}/{len(tp_df)}: {os.path.basename(fpath)}", end='\r')

    print(f"\n\nExtraction Complete. Total vectors: {len(features_list)}")
    
    # 4. Calculate Statistics
    feature_matrix = np.array(features_list)
    
    print("Calculating Mahalanobis Parameters...")
    mean = np.mean(feature_matrix, axis=0)
    cov = np.cov(feature_matrix, rowvar=False)
    
    # Regularization (Crucial for Inversion stability)
    # Adds a tiny jitter to the diagonal
    reg_cov = cov + np.eye(feature_matrix.shape[1]) * 1e-5
    
    try:
        precision_matrix = np.linalg.inv(reg_cov)
        print("Matrix Inversion Successful.")
    except np.linalg.LinAlgError:
        print(f"{Fore.RED}Matrix Singularity Error. Try increasing regularization.{Style.RESET_ALL}")
        sys.exit(1)

    # 5. Save
    output_data = {
        'mean': mean,
        'precision_matrix': precision_matrix,
        'n_samples': len(features_list)
    }
    
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(output_data, f)
        
    print(f"{Fore.GREEN}SUCCESS! Saved parameters to {OUTPUT_FILE}{Style.RESET_ALL}")
    print("Next Step: Copy this .pkl file to your Raspberry Pi.")
