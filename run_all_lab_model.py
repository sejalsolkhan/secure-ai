"""
CoughGuard - AudioSet Model Inference
=====================================

Runs inference on either Coswara or ESC-50 dataset.
Outputs results to timestamped CSV files in outputs directory.

Usage:
    python run_dataset_inference.py coswara
    python run_dataset_inference.py esc50
    python run_dataset_inference.py both
"""

import sys
sys.path.insert(0, '.')

import os
import csv
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import pandas as pd
from models.mn.model import get_model
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Dataset paths
COSWARA_PATH = "datasets/Coswara-Data/Extracted_data"
ESC50_PATH = "datasets/ESC-50"
ESC50_AUDIO_PATH = "datasets/ESC-50/audio"
ESC50_META_PATH = "datasets/ESC-50/meta/esc50.csv"

# Model path
#MODEL_PATH = "models/pretrained/mn10_as.pt"
MODEL_PATH = "models/mn/mn10_mydata_epoch_13.pt"

# Output directory
OUTPUT_DIR = "outputs"

# Preprocessing parameters (EfficientAT defaults)
"""
SAMPLE_RATE = 32000
N_FFT = 800 
HOP_LENGTH = 320
N_MELS = 128
"""
SAMPLE_RATE = 32000
N_FFT = 1024
HOP_LENGTH = 320
N_MELS = 128
WIN_LENGTH = 800
FMIN = 0.0               # Minimum frequency for mel filterbank
FMAX = 15000             # Maximum frequency


# lab class IDs
NUM_CLASSES = 2          # Lab model has 22 classes
COUGH_CLASS_ID = 0       # Cough is index 0 in lab's model

# Lab's class labels
LAB_LABELS = {
    0: "Cough",
    1: "Not Cough",
}

# Coswara cough files (ground truth = cough)
COSWARA_COUGH_FILES = ['cough-heavy.wav', 'cough-shallow.wav']

# ESC-50 cough category
ESC50_COUGH_CATEGORY = 'coughing'

# Detection threshold
COUGH_THRESHOLD = 0.7

# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess_audio(file_path: str):
    """
    Convert audio file to mel-spectrogram
    
    Pipeline:
    1. Load audio and convert to mono
    2. Resample to 32kHz (model's expected sample rate)
    3. Apply preemphasis filter (boosts high frequencies)
    4. Compute STFT (Short-Time Fourier Transform)
    5. Convert to power spectrogram
    6. Apply mel filterbank
    7. Convert to log scale
    8. Normalize using lab's formula: (x + 4.5) / 5.0
    
    Returns:
        mel_spec: Tensor of shape (1, n_mels, n_frames) ready for model
    """
    
    # =========================================================================
    # STEP 1: Load audio file
    # =========================================================================
    # torchaudio.load returns: (waveform, sample_rate)
    # waveform shape: (num_channels, num_samples)
    waveform, sr = torchaudio.load(file_path)
    
    # =========================================================================
    # STEP 2: Convert to mono (single channel)
    # =========================================================================
    # If stereo (2 channels), average them to get mono
    # Model expects single-channel audio
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    # waveform shape is now: (1, num_samples)
    
    # =========================================================================
    # STEP 3: Resample to 32kHz
    # =========================================================================
    # Model was trained on 32kHz audio
    # If audio has different sample rate, we must resample
    # ESC-50 is 44.1kHz, Coswara is 48kHz - both need resampling
    if sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    
    # =========================================================================
    # STEP 4: Apply preemphasis filter
    # =========================================================================
    # Preemphasis boosts high frequencies relative to low frequencies
    # This helps the model better distinguish speech/cough sounds
    # 
    # Formula: y[n] = x[n] - 0.97 * x[n-1]
    # Implemented as convolution with kernel [-0.97, 1]
    #
    # Why? Human hearing is more sensitive to high frequencies,
    # and this filter compensates for that, improving feature extraction
    preemphasis_coefficient = torch.tensor([[[-.97, 1]]])
    waveform = torch.nn.functional.conv1d(
        waveform.unsqueeze(0),  # Add batch dim: (1, 1, num_samples)
        preemphasis_coefficient
    ).squeeze(0)  # Remove batch dim: (1, num_samples)
    
    # =========================================================================
    # STEP 5: Compute STFT (Short-Time Fourier Transform)
    # =========================================================================
    # STFT breaks audio into overlapping windows and computes FFT on each
    # This gives us frequency content over time
    #
    # Parameters:
    # - n_fft=1024: Each FFT analyzes 1024 samples (~32ms at 32kHz)
    # - win_length=800: But we only use 800 samples per window (25ms)
    # - hop_length=320: Move 320 samples (10ms) between windows
    # - window: Hann window smooths the edges to reduce artifacts
    # - center=True: Pad signal so first window is centered at t=0
    # - return_complex=False: Return real/imag parts separately
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
    # stft shape: (1, n_fft//2 + 1, n_frames, 2)
    # The last dimension is [real, imaginary] parts
    
    # =========================================================================
    # STEP 6: Convert to power spectrogram
    # =========================================================================
    # Power = real^2 + imag^2
    # This gives us the energy at each frequency/time
    power_spec = (stft ** 2).sum(dim=-1)
    # power_spec shape: (1, n_fft//2 + 1, n_frames) = (1, 513, n_frames)
    
    # =========================================================================
    # STEP 7: Apply mel filterbank
    # =========================================================================
    # Mel scale mimics human hearing (we hear low frequencies in finer detail)
    # The filterbank converts 513 FFT bins to 128 mel bins
    #
    # get_mel_banks returns triangular filters that overlap in frequency
    # Each filter picks out energy in a specific frequency range
    mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
        N_MELS,           # Number of mel bins (128)
        N_FFT,            # FFT size (1024)
        SAMPLE_RATE,      # Sample rate (32000)
        FMIN,             # Min frequency (0 Hz)
        FMAX,             # Max frequency (15000 Hz)
        vtln_low=100.0,   # Vocal tract length normalization params
        vtln_high=-500.,  # (these are standard Kaldi defaults)
        vtln_warp_factor=1.0
    )
    # mel_basis shape: (128, 513)
    
    # Pad mel_basis to match power_spec dimensions
    # (adds one zero column to make it 128 x 514, matching n_fft//2 + 1 = 513... 
    #  actually this pads to handle edge cases in the FFT output)
    mel_basis = torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0)
    
    # Matrix multiply: mel_basis (128 x 514) @ power_spec (514 x n_frames)
    # Result: mel spectrogram (128 x n_frames)
    melspec = torch.matmul(mel_basis, power_spec.squeeze(0))
    # melspec shape: (128, n_frames)
    
    # =========================================================================
    # STEP 8: Convert to log scale (decibels)
    # =========================================================================
    # Human hearing perceives loudness logarithmically
    # Add small epsilon (0.00001) to avoid log(0)
    melspec = (melspec + 0.00001).log()
    
    # =========================================================================
    # STEP 9: Normalize using lab's formula
    # =========================================================================
    # Lab uses "fast normalization": (x + 4.5) / 5.0
    # This shifts and scales the values to roughly [0, 1] range
    #
    # Why 4.5 and 5.0? These are empirically chosen constants that work well
    # for the typical range of log-mel values in speech/audio
    #
    # NOTE: This is DIFFERENT from standard mean/std normalization!
    # Using wrong normalization will hurt model performance
    melspec = (melspec + 4.5) / 5.0
    
    # =========================================================================
    # STEP 10: Add batch dimension for model input
    # =========================================================================
    # Model expects input shape: (batch_size, channel, n_mels, n_frames)
    melspec = melspec.unsqueeze(0)
    melspec = melspec.unsqueeze(0)
    # Final shape: (1, 1, 128, n_frames)
    
    return melspec

# =============================================================================
# MODEL
# =============================================================================

def load_model():
    """Load the EfficientAT AudioSet model."""
    
    model = get_model(
        num_classes=NUM_CLASSES,
        pretrained_name=None,
        width_mult=1.0,
        head_type="mlp",
        se_dims='c'
    )
    
    
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model

# =============================================================================
# INFERENCE
# =============================================================================

def run_inference(model, file_path: str):
    """Run inference on a single audio file."""
    
    mel_spec = preprocess_audio(file_path)
    
    with torch.no_grad():
        logits, _ = model(mel_spec)
        # Use SOFTMAX for multi-class (not sigmoid like AudioSet)
        probs = F.softmax(logits, dim=1)[0]
    
    cough_prob = probs[COUGH_CLASS_ID].item()
    
    # Get predictions
    top_indices = probs.argsort(descending=True)[:2]
    top_preds = []
    for idx in top_indices:
        label = LAB_LABELS.get(idx.item(), f"Class {idx.item()}")
        top_preds.append(f"{label} ({probs[idx].item():.1%})")
    
    # All class probabilities
    class_probs = {
        'cough': probs[0].item(),
        'not_cough': probs[1].item(),
    }
    
    return cough_prob, top_preds,class_probs 

# =============================================================================
# DATASET PROCESSING
# =============================================================================

def process_coswara(model):
    """Process Coswara dataset."""
    
    results = []
    
    if not os.path.exists(COSWARA_PATH):
        print(f"❌ Coswara dataset not found: {COSWARA_PATH}")
        return results
    
    date_folders = [d for d in os.listdir(COSWARA_PATH) 
                    if os.path.isdir(os.path.join(COSWARA_PATH, d))]
    
    print(f"   Found {len(date_folders)} date folder(s)")
    
    for date_folder in sorted(date_folders):
        date_path = os.path.join(COSWARA_PATH, date_folder)
        
        participants = [p for p in os.listdir(date_path)
                       if os.path.isdir(os.path.join(date_path, p))]
        
        print(f"   Processing {date_folder}: {len(participants)} participant(s)")
        
        for participant_id in participants:
            participant_path = os.path.join(date_path, participant_id)
            wav_files = [f for f in os.listdir(participant_path) if f.endswith('.wav')]
            
            for wav_file in wav_files:
                file_path = os.path.join(participant_path, wav_file)
                is_cough_file = wav_file in COSWARA_COUGH_FILES
                
                try:
                    cough_prob, top_preds, resp_probs = run_inference(model, file_path)
                    model_says_cough = cough_prob > COUGH_THRESHOLD
                    
                    if is_cough_file and model_says_cough:
                        result = "TP"
                    elif not is_cough_file and not model_says_cough:
                        result = "TN"
                    elif not is_cough_file and model_says_cough:
                        result = "FP"
                    else:
                        result = "FN"
                    
                    results.append({
                        'dataset': 'Coswara',
                        'file_path': file_path,
                        'filename': wav_file,
                        'category': 'cough' if is_cough_file else wav_file.replace('.wav', ''),
                        'ground_truth': 'cough' if is_cough_file else 'not_cough',
                        'is_cough_ground_truth': is_cough_file,
                        'cough_probability': round(cough_prob, 4),
                        'model_prediction': 'cough' if model_says_cough else 'not_cough',
                        'result': result,
                        'top_prediction': top_preds[0] if top_preds else '',
                        'top_3_predictions': ' | '.join(top_preds),
                        'prob_cough': round(resp_probs['cough'], 4),
                        'prob_not_cough': round(resp_probs['not_cough'], 4),
                    })
                    
                except Exception as e:
                    print(f"      ❌ Error: {wav_file}: {e}")
    
    return results


def process_esc50(model):
    """Process ESC-50 dataset."""
    
    results = []
    
    if not os.path.exists(ESC50_META_PATH):
        print(f"❌ ESC-50 metadata not found: {ESC50_META_PATH}")
        return results
    
    if not os.path.exists(ESC50_AUDIO_PATH):
        print(f"❌ ESC-50 audio folder not found: {ESC50_AUDIO_PATH}")
        return results
    
    # Load metadata
    meta = pd.read_csv(ESC50_META_PATH)
    print(f"   Found {len(meta)} audio files in ESC-50")
    
    # Show category breakdown
    print(f"   Categories: {meta['category'].nunique()} unique")
    cough_count = len(meta[meta['category'] == ESC50_COUGH_CATEGORY])
    print(f"   Cough samples: {cough_count}")
    
    for idx, row in meta.iterrows():
        file_path = os.path.join(ESC50_AUDIO_PATH, row['filename'])
        
        if not os.path.exists(file_path):
            continue
        
        is_cough_file = row['category'] == ESC50_COUGH_CATEGORY
        
        try:
            cough_prob, top_preds, resp_probs = run_inference(model, file_path)
            model_says_cough = cough_prob > COUGH_THRESHOLD
            
            if is_cough_file and model_says_cough:
                result = "TP"
            elif not is_cough_file and not model_says_cough:
                result = "TN"
            elif not is_cough_file and model_says_cough:
                result = "FP"
            else:
                result = "FN"
            
            results.append({
                'dataset': 'ESC-50',
                'file_path': file_path,
                'filename': row['filename'],
                'category': row['category'],
                'ground_truth': 'cough' if is_cough_file else 'not_cough',
                'is_cough_ground_truth': is_cough_file,
                'cough_probability': round(cough_prob, 4),
                'model_prediction': 'cough' if model_says_cough else 'not_cough',
                'result': result,
                'top_prediction': top_preds[0] if top_preds else '',
                'top_3_predictions': ' | '.join(top_preds),
                'prob_cough': round(resp_probs['cough'], 4),
                'prob_not_cough': round(resp_probs['not_cough'], 4),
                'esc50_fold': row['fold'],
                'esc50_target': row['target'],
            })
            
        except Exception as e:
            print(f"      ❌ Error: {row['filename']}: {e}")
        
        # Progress
        if (idx + 1) % 500 == 0:
            print(f"   Processed {idx + 1}/{len(meta)} files")
    
    return results

# =============================================================================
# CSV OUTPUT
# =============================================================================

def save_results_to_csv(results, dataset_name, timestamp):
    """Save all results, false positives, false negatives, and summary to CSV files."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not results:
        print("No results to save!")
        return
    
    # Calculate metrics
    total = len(results)
    tp = sum(1 for r in results if r['result'] == 'TP')
    tn = sum(1 for r in results if r['result'] == 'TN')
    fp = sum(1 for r in results if r['result'] == 'FP')
    fn = sum(1 for r in results if r['result'] == 'FN')
    
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 1. Save ALL results
    all_results_file = os.path.join(OUTPUT_DIR, f"all_results_{dataset_name}_{timestamp}.csv")
    fieldnames = list(results[0].keys())
    
    with open(all_results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"✓ All results: {all_results_file}")
    
    # 2. Save FALSE NEGATIVES
    false_negatives = [r for r in results if r['result'] == 'FN']
    if false_negatives:
        fn_file = os.path.join(OUTPUT_DIR, f"false_negatives_{dataset_name}_{timestamp}.csv")
        with open(fn_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(false_negatives)
        print(f"✓ False negatives ({len(false_negatives)}): {fn_file}")
    
    # 3. Save FALSE POSITIVES
    false_positives = [r for r in results if r['result'] == 'FP']
    if false_positives:
        fp_file = os.path.join(OUTPUT_DIR, f"false_positives_{dataset_name}_{timestamp}.csv")
        with open(fp_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(false_positives)
        print(f"✓ False positives ({len(false_positives)}): {fp_file}")
    
    # 4. Save SUMMARY
    summary_file = os.path.join(OUTPUT_DIR, f"summary_{dataset_name}_{timestamp}.csv")
    
    # Cough file analysis
    cough_results = [r for r in results if r['is_cough_ground_truth']]
    non_cough_results = [r for r in results if not r['is_cough_ground_truth']]
    
    avg_cough_prob_cough = sum(r['cough_probability'] for r in cough_results) / len(cough_results) if cough_results else 0
    avg_cough_prob_noncough = sum(r['cough_probability'] for r in non_cough_results) / len(non_cough_results) if non_cough_results else 0
    
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header section
        writer.writerow(['=== INFERENCE SUMMARY ===', ''])
        writer.writerow(['Dataset', dataset_name])
        writer.writerow(['Timestamp', timestamp])
        writer.writerow(['Model', MODEL_PATH])
        writer.writerow(['Threshold', COUGH_THRESHOLD])
        writer.writerow(['', ''])
        
        # Confusion matrix
        writer.writerow(['=== CONFUSION MATRIX ===', ''])
        writer.writerow(['', 'Predicted Cough', 'Predicted Not-Cough'])
        writer.writerow(['Actual Cough', tp, fn])
        writer.writerow(['Actual Not-Cough', fp, tn])
        writer.writerow(['', ''])
        
        # Metrics
        writer.writerow(['=== METRICS ===', ''])
        writer.writerow(['Total Files', total])
        writer.writerow(['True Positives (TP)', tp])
        writer.writerow(['True Negatives (TN)', tn])
        writer.writerow(['False Positives (FP)', fp])
        writer.writerow(['False Negatives (FN)', fn])
        writer.writerow(['Accuracy', f"{accuracy:.4f}"])
        writer.writerow(['Precision', f"{precision:.4f}"])
        writer.writerow(['Recall', f"{recall:.4f}"])
        writer.writerow(['F1 Score', f"{f1:.4f}"])
        writer.writerow(['', ''])
        
        # Cough file analysis
        writer.writerow(['=== COUGH FILE ANALYSIS ===', ''])
        writer.writerow(['Total Cough Files', len(cough_results)])
        writer.writerow(['Detected as Cough', tp])
        writer.writerow(['Detection Rate', f"{tp/len(cough_results):.4f}" if cough_results else "N/A"])
        writer.writerow(['Avg Cough Probability', f"{avg_cough_prob_cough:.4f}"])
        writer.writerow(['', ''])
        
        # Non-cough file analysis
        writer.writerow(['=== NON-COUGH FILE ANALYSIS ===', ''])
        writer.writerow(['Total Non-Cough Files', len(non_cough_results)])
        writer.writerow(['False Positives', fp])
        writer.writerow(['False Positive Rate', f"{fp/len(non_cough_results):.4f}" if non_cough_results else "N/A"])
        writer.writerow(['Avg Cough Probability', f"{avg_cough_prob_noncough:.4f}"])
        
        # Category breakdown for ESC-50 or file type breakdown for Coswara
        if dataset_name.lower() == 'esc50':
            writer.writerow(['', ''])
            writer.writerow(['=== FALSE POSITIVES BY CATEGORY ===', ''])
            fp_by_category = {}
            for r in false_positives:
                cat = r['category']
                fp_by_category[cat] = fp_by_category.get(cat, 0) + 1
            for cat, count in sorted(fp_by_category.items(), key=lambda x: -x[1]):
                writer.writerow([cat, count])
        
        elif dataset_name.lower() == 'coswara':
            writer.writerow(['', ''])
            writer.writerow(['=== COUGH PROBABILITY BY FILE TYPE ===', ''])
            file_types = {}
            for r in non_cough_results:
                ft = r['filename']
                if ft not in file_types:
                    file_types[ft] = []
                file_types[ft].append(r['cough_probability'])
            for ft, probs in sorted(file_types.items()):
                avg = sum(probs) / len(probs)
                writer.writerow([ft, f"{avg:.4f}"])
    
    print(f"✓ Summary: {summary_file}")


def print_summary(results, dataset_name):
    """Print summary to console."""
    
    print("\n" + "=" * 70)
    print(f"RESULTS SUMMARY - {dataset_name}")
    print("=" * 70)
    
    total = len(results)
    if total == 0:
        print("No results!")
        return
    
    tp = sum(1 for r in results if r['result'] == 'TP')
    tn = sum(1 for r in results if r['result'] == 'TN')
    fp = sum(1 for r in results if r['result'] == 'FP')
    fn = sum(1 for r in results if r['result'] == 'FN')
    
    print(f"\nTotal files: {total}")
    
    print(f"\nConfusion Matrix:")
    print(f"                      Predicted")
    print(f"                   Cough    Not-Cough")
    print(f"  Actual Cough      {tp:4d}       {fn:4d}    (TP / FN)")
    print(f"  Actual Other      {fp:4d}       {tn:4d}    (FP / TN)")
    
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1 Score:  {f1:.1%}")
    
    # Cough file analysis
    cough_results = [r for r in results if r['is_cough_ground_truth']]
    if cough_results:
        avg_prob = sum(r['cough_probability'] for r in cough_results) / len(cough_results)
        detected = sum(1 for r in cough_results if r['result'] == 'TP')
        print(f"\nCough Files: {len(cough_results)} total, {detected} detected ({100*detected/len(cough_results):.1f}%)")
        print(f"  Avg cough probability: {avg_prob:.1%}")
    
    # Non-cough analysis
    non_cough_results = [r for r in results if not r['is_cough_ground_truth']]
    if non_cough_results:
        avg_prob = sum(r['cough_probability'] for r in non_cough_results) / len(non_cough_results)
        fp_count = sum(1 for r in non_cough_results if r['result'] == 'FP')
        print(f"\nNon-Cough Files: {len(non_cough_results)} total, {fp_count} false positives ({100*fp_count/len(non_cough_results):.1f}%)")
        print(f"  Avg cough probability: {avg_prob:.1%}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run cough detection inference on Coswara or ESC-50 dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_dataset_inference.py coswara
    python run_dataset_inference.py esc50
    python run_dataset_inference.py both
        """
    )
    parser.add_argument('dataset', choices=['coswara', 'esc50', 'both'],
                        help='Which dataset to process')
    
    args = parser.parse_args()
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("CoughGuard - AudioSet Model Inference")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Timestamp: {timestamp}")
    print("=" * 70)
    
    # Check model
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model not found: {MODEL_PATH}")
        return
    
    # Load model
    print("\n1. Loading model...")
    model = load_model()
    print("   ✓ Model loaded!")
    
    # Process datasets
    if args.dataset in ['coswara', 'both']:
        print("\n2. Processing Coswara dataset...")
        coswara_results = process_coswara(model)
        if coswara_results:
            print_summary(coswara_results, "Coswara")
            save_results_to_csv(coswara_results, "coswara", timestamp)
    
    if args.dataset in ['esc50', 'both']:
        print("\n3. Processing ESC-50 dataset...")
        esc50_results = process_esc50(model)
        if esc50_results:
            print_summary(esc50_results, "ESC-50")
            save_results_to_csv(esc50_results, "esc50", timestamp)
    
    if args.dataset == 'both':
        all_results = []
        if 'coswara_results' in dir() and coswara_results:
            all_results.extend(coswara_results)
        if 'esc50_results' in dir() and esc50_results:
            all_results.extend(esc50_results)
        if all_results:
            print_summary(all_results, "Combined")
            save_results_to_csv(all_results, "combined", timestamp)
    
    print("\n" + "=" * 70)
    print(f"DONE! CSV files saved to '{OUTPUT_DIR}' folder.")
    print("=" * 70)


if __name__ == "__main__":
    main()
