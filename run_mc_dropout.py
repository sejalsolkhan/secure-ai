"""
CoughGuard - Monte Carlo Dropout Uncertainty Quantification
============================================================

Runs MC Dropout inference on ESC-50 dataset to quantify prediction uncertainty.

What this script does:
1. Loads audio files from ESC-50 dataset
2. Runs the model 20 times per file with dropout enabled
3. Computes uncertainty metrics from the variation in predictions
4. Analyzes correlation between uncertainty and errors
5. Outputs CSV with uncertainty scores + summary analysis

Usage:
    python run_mc_dropout_esc50.py

Requirements:
    - ESC-50 dataset in datasets/ESC-50/
    - Lab's model in models/mn/
"""

import sys
sys.path.insert(0, '.')

import os
import csv
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import pandas as pd
from models.mn.model import get_model
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION (Matching run_all_lab_model.py exactly)
# =============================================================================

# Dataset paths
COSWARA_PATH = "datasets/Coswara-Data/Extracted_data"
ESC50_PATH = "datasets/ESC-50"
ESC50_AUDIO_PATH = "datasets/ESC-50/audio"
ESC50_META_PATH = "datasets/ESC-50/meta/esc50.csv"

# Model path - SAME AS run_all_lab_model.py
MODEL_PATH = "models/mn/mn10_mydata_epoch_13.pt"

# Output directory
OUTPUT_DIR = "outputs"

# =============================================================================
# PREPROCESSING PARAMETERS (Matching run_all_lab_model.py exactly)
# =============================================================================

SAMPLE_RATE = 32000
N_FFT = 1024
HOP_LENGTH = 320
N_MELS = 128
WIN_LENGTH = 800
FMIN = 0.0
FMAX = 15000

# =============================================================================
# MODEL CONFIGURATION (Matching run_all_lab_model.py exactly)
# =============================================================================

NUM_CLASSES = 2          # Lab model has 2 classes
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

# Detection threshold - SAME AS run_all_lab_model.py
COUGH_THRESHOLD = 0.7

# =============================================================================
# MC DROPOUT CONFIGURATION
# =============================================================================

MC_DROPOUT_PASSES = 20   # Number of forward passes
DROPOUT_RATE = 0.2       # Model's dropout rate (from model.py line 82)

# =============================================================================
# PREPROCESSING (Same as run_all_lab_model.py)
# =============================================================================

def preprocess_audio(file_path: str):
    """
    Convert audio file to mel-spectrogram using lab's exact method.
    
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
    
    # STEP 1: Load audio file
    waveform, sr = torchaudio.load(file_path)
    
    # STEP 2: Convert to mono (single channel)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # STEP 3: Resample to 32kHz
    if sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    
    # STEP 4: Apply preemphasis filter
    preemphasis_coefficient = torch.tensor([[[-.97, 1]]])
    waveform = torch.nn.functional.conv1d(
        waveform.unsqueeze(0),
        preemphasis_coefficient
    ).squeeze(0)
    
    # STEP 5: Compute STFT
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
    
    # STEP 6: Convert to power spectrogram
    power_spec = (stft ** 2).sum(dim=-1)
    
    # STEP 7: Apply mel filterbank
    mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
        N_MELS,
        N_FFT,
        SAMPLE_RATE,
        FMIN,
        FMAX,
        vtln_low=100.0,
        vtln_high=-500.,
        vtln_warp_factor=1.0
    )
    mel_basis = torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0)
    
    # Apply mel filterbank
    melspec = torch.matmul(mel_basis, power_spec.squeeze(0))
    
    # STEP 8: Log scale
    melspec = (melspec + 0.00001).log()
    
    # STEP 9: Lab's normalization
    melspec = (melspec + 4.5) / 5.0
    
    # STEP 10: Add batch and channel dimensions
    # Model expects input shape: (batch_size, channel, n_mels, n_frames)
    melspec = melspec.unsqueeze(0)  # Add channel dim: (1, 128, n_frames)
    melspec = melspec.unsqueeze(0)  # Add batch dim: (1, 1, 128, n_frames)
    
    return melspec

# =============================================================================
# MODEL LOADING (Same as run_all_lab_model.py)
# =============================================================================

def load_model():
    """Load the lab's 2-class cough detection model."""
    
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
    
    return model

# =============================================================================
# MC DROPOUT FUNCTIONS
# =============================================================================

def enable_mc_dropout(model):
    """
    Enable dropout during inference for Monte Carlo Dropout.
    
    Key insight: We want BatchNorm in eval mode (use population stats)
    but Dropout in train mode (randomly drop neurons).
    """
    model.eval()  # Set everything to eval mode first
    
    # Then selectively enable dropout layers
    dropout_count = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()  # Enable dropout
            dropout_count += 1
    
    return model, dropout_count


def mc_dropout_inference(model, mel_spec, n_passes=20):
    """
    Run Monte Carlo Dropout inference.
    
    Args:
        model: The neural network with dropout
        mel_spec: Preprocessed audio (mel spectrogram)
        n_passes: Number of stochastic forward passes
    
    Returns:
        Dictionary with predictions and uncertainty metrics
    """
    
    # Enable MC Dropout
    model, _ = enable_mc_dropout(model)
    
    # Collect predictions from multiple passes
    all_probs = []
    
    with torch.no_grad():
        for i in range(n_passes):
            logits, _ = model(mel_spec)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)
    
    # Stack: shape (n_passes, batch_size, num_classes)
    all_probs = torch.stack(all_probs, dim=0)
    
    # ==========================================================================
    # COMPUTE UNCERTAINTY METRICS
    # ==========================================================================
    
    # 1. Mean prediction (average across MC passes)
    mean_probs = all_probs.mean(dim=0)  # Shape: (batch_size, num_classes)
    
    # 2. Variance of predictions (measures spread)
    var_probs = all_probs.var(dim=0)    # Shape: (batch_size, num_classes)
    
    # 3. Predictive Entropy (total uncertainty)
    #    H[y|x,D] = -sum(p * log(p))
    #    High entropy = model is unsure between classes
    predictive_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)
    
    # 4. Expected Entropy (aleatoric uncertainty - data noise)
    #    E[H[y|x,w]] = average entropy of individual predictions
    individual_entropies = -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=2)
    expected_entropy = individual_entropies.mean(dim=0)
    
    # 5. Mutual Information (epistemic uncertainty - model uncertainty)
    #    I[y,w|x,D] = Predictive Entropy - Expected Entropy
    #    High MI = model is uncertain due to lack of knowledge
    mutual_information = predictive_entropy - expected_entropy
    
    # 6. Confidence (max probability from mean prediction)
    confidence = mean_probs.max(dim=1).values
    
    # 7. Standard deviation of cough probability across passes
    cough_probs_all = all_probs[:, :, COUGH_CLASS_ID]  # Shape: (n_passes, batch_size)
    cough_prob_std = cough_probs_all.std(dim=0)
    
    # 8. Min and max cough probability across passes (shows range)
    cough_prob_min = cough_probs_all.min(dim=0).values
    cough_prob_max = cough_probs_all.max(dim=0).values
    
    return {
        'mean_probs': mean_probs,
        'all_probs': all_probs,
        'variance': var_probs,
        'predictive_entropy': predictive_entropy,
        'expected_entropy': expected_entropy,
        'mutual_information': mutual_information,
        'confidence': confidence,
        'cough_prob_std': cough_prob_std,
        'cough_prob_min': cough_prob_min,
        'cough_prob_max': cough_prob_max,
    }

# =============================================================================
# DATASET PROCESSING
# =============================================================================

def process_esc50_with_mc_dropout(model):
    """Process ESC-50 dataset with MC Dropout."""
    
    results = []
    
    if not os.path.exists(ESC50_META_PATH):
        print(f"❌ ESC-50 metadata not found: {ESC50_META_PATH}")
        return results
    
    # Load metadata
    meta = pd.read_csv(ESC50_META_PATH)
    total_files = len(meta)
    
    print(f"   Found {total_files} audio files")
    print(f"   Running {MC_DROPOUT_PASSES} MC Dropout passes per file")
    print(f"   Total forward passes: {total_files * MC_DROPOUT_PASSES}")
    
    for idx, row in meta.iterrows():
        file_path = os.path.join(ESC50_AUDIO_PATH, row['filename'])
        
        if not os.path.exists(file_path):
            continue
        
        is_cough_file = row['category'] == ESC50_COUGH_CATEGORY
        
        try:
            # Preprocess
            mel_spec = preprocess_audio(file_path)
            
            # Run MC Dropout inference
            mc_results = mc_dropout_inference(model, mel_spec, n_passes=MC_DROPOUT_PASSES)
            
            # Extract results
            mean_probs = mc_results['mean_probs'][0]
            cough_prob = mean_probs[COUGH_CLASS_ID].item()
            
            # Uncertainty metrics
            pred_entropy = mc_results['predictive_entropy'][0].item()
            mutual_info = mc_results['mutual_information'][0].item()
            aleatoric = mc_results['expected_entropy'][0].item()
            confidence = mc_results['confidence'][0].item()
            cough_std = mc_results['cough_prob_std'][0].item()
            variance = mc_results['variance'][0, COUGH_CLASS_ID].item()
            cough_min = mc_results['cough_prob_min'][0].item()
            cough_max = mc_results['cough_prob_max'][0].item()
            
            # Classification (using same threshold as run_all_lab_model.py)
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
                'filename': row['filename'],
                'file_path': file_path,
                'category': row['category'],
                'ground_truth': 'cough' if is_cough_file else 'not_cough',
                'is_cough': is_cough_file,
                
                # Predictions
                'cough_probability': round(cough_prob, 4),
                'model_prediction': 'cough' if model_says_cough else 'not_cough',
                'result': result,
                'confidence': round(confidence, 4),
                
                # Uncertainty metrics
                'predictive_entropy': round(pred_entropy, 4),
                'epistemic_uncertainty': round(mutual_info, 4),
                'aleatoric_uncertainty': round(aleatoric, 4),
                'cough_prob_std': round(cough_std, 4),
                'cough_prob_variance': round(variance, 6),
                'cough_prob_min': round(cough_min, 4),
                'cough_prob_max': round(cough_max, 4),
                
                # ESC-50 metadata
                'esc50_fold': row['fold'],
                'esc50_target': row['target'],
            })
            
        except Exception as e:
            print(f"   ❌ Error: {row['filename']}: {e}")
        
        # Progress
        if (idx + 1) % 100 == 0:
            print(f"   Processed {idx + 1}/{total_files} files")
    
    return results


def process_coswara_with_mc_dropout(model):
    """Process Coswara dataset with MC Dropout."""
    
    results = []
    
    if not os.path.exists(COSWARA_PATH):
        print(f"❌ Coswara dataset not found: {COSWARA_PATH}")
        return results
    
    date_folders = [d for d in os.listdir(COSWARA_PATH) 
                    if os.path.isdir(os.path.join(COSWARA_PATH, d))]
    
    print(f"   Found {len(date_folders)} date folder(s)")
    
    file_count = 0
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
                    # Preprocess
                    mel_spec = preprocess_audio(file_path)
                    
                    # Run MC Dropout inference
                    mc_results = mc_dropout_inference(model, mel_spec, n_passes=MC_DROPOUT_PASSES)
                    
                    # Extract results
                    mean_probs = mc_results['mean_probs'][0]
                    cough_prob = mean_probs[COUGH_CLASS_ID].item()
                    
                    # Uncertainty metrics
                    pred_entropy = mc_results['predictive_entropy'][0].item()
                    mutual_info = mc_results['mutual_information'][0].item()
                    aleatoric = mc_results['expected_entropy'][0].item()
                    confidence = mc_results['confidence'][0].item()
                    cough_std = mc_results['cough_prob_std'][0].item()
                    variance = mc_results['variance'][0, COUGH_CLASS_ID].item()
                    cough_min = mc_results['cough_prob_min'][0].item()
                    cough_max = mc_results['cough_prob_max'][0].item()
                    
                    # Classification
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
                        'filename': wav_file,
                        'file_path': file_path,
                        'category': 'cough' if is_cough_file else wav_file.replace('.wav', ''),
                        'ground_truth': 'cough' if is_cough_file else 'not_cough',
                        'is_cough': is_cough_file,
                        
                        # Predictions
                        'cough_probability': round(cough_prob, 4),
                        'model_prediction': 'cough' if model_says_cough else 'not_cough',
                        'result': result,
                        'confidence': round(confidence, 4),
                        
                        # Uncertainty metrics
                        'predictive_entropy': round(pred_entropy, 4),
                        'epistemic_uncertainty': round(mutual_info, 4),
                        'aleatoric_uncertainty': round(aleatoric, 4),
                        'cough_prob_std': round(cough_std, 4),
                        'cough_prob_variance': round(variance, 6),
                        'cough_prob_min': round(cough_min, 4),
                        'cough_prob_max': round(cough_max, 4),
                        
                        # Coswara metadata
                        'participant_id': participant_id,
                        'date_folder': date_folder,
                    })
                    
                    file_count += 1
                    
                except Exception as e:
                    print(f"      ❌ Error: {wav_file}: {e}")
    
    print(f"   Total files processed: {file_count}")
    return results

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_uncertainty_vs_errors(results):
    """Analyze correlation between uncertainty and errors."""
    
    print("\n" + "=" * 70)
    print("UNCERTAINTY vs ERROR ANALYSIS")
    print("=" * 70)
    
    # Separate correct and incorrect predictions
    correct = [r for r in results if r['result'] in ['TP', 'TN']]
    incorrect = [r for r in results if r['result'] in ['FP', 'FN']]
    
    if not correct or not incorrect:
        print("Not enough data for analysis")
        return {}
    
    # Calculate mean uncertainty for correct vs incorrect
    metrics = ['predictive_entropy', 'epistemic_uncertainty', 'aleatoric_uncertainty', 'cough_prob_std']
    
    analysis = {}
    
    print(f"\nCorrect predictions: {len(correct)}")
    print(f"Incorrect predictions: {len(incorrect)}")
    
    print("\n--- Mean Uncertainty by Correctness ---")
    print(f"{'Metric':<25} {'Correct':>12} {'Incorrect':>12} {'Diff':>10}")
    print("-" * 60)
    
    for metric in metrics:
        correct_mean = np.mean([r[metric] for r in correct])
        incorrect_mean = np.mean([r[metric] for r in incorrect])
        diff = incorrect_mean - correct_mean
        diff_pct = (diff / correct_mean * 100) if correct_mean > 0 else 0
        
        print(f"{metric:<25} {correct_mean:>12.4f} {incorrect_mean:>12.4f} {diff_pct:>+9.1f}%")
        
        analysis[f'{metric}_correct'] = correct_mean
        analysis[f'{metric}_incorrect'] = incorrect_mean
        analysis[f'{metric}_diff_pct'] = diff_pct
    
    # Analyze by result type
    print("\n--- Mean Uncertainty by Result Type ---")
    for result_type in ['TP', 'TN', 'FP', 'FN']:
        subset = [r for r in results if r['result'] == result_type]
        if subset:
            mean_entropy = np.mean([r['predictive_entropy'] for r in subset])
            mean_epistemic = np.mean([r['epistemic_uncertainty'] for r in subset])
            mean_std = np.mean([r['cough_prob_std'] for r in subset])
            print(f"{result_type}: n={len(subset):4d}, entropy={mean_entropy:.4f}, epistemic={mean_epistemic:.4f}, std={mean_std:.4f}")
    
    # Check if high uncertainty predicts errors
    print("\n--- Uncertainty Distribution ---")
    
    all_entropy = [r['predictive_entropy'] for r in results]
    median_entropy = np.median(all_entropy)
    
    high_uncertainty = [r for r in results if r['predictive_entropy'] > median_entropy]
    low_uncertainty = [r for r in results if r['predictive_entropy'] <= median_entropy]
    
    high_error_rate = len([r for r in high_uncertainty if r['result'] in ['FP', 'FN']]) / len(high_uncertainty) if high_uncertainty else 0
    low_error_rate = len([r for r in low_uncertainty if r['result'] in ['FP', 'FN']]) / len(low_uncertainty) if low_uncertainty else 0
    
    print(f"\nMedian predictive entropy: {median_entropy:.4f}")
    print(f"High uncertainty (>{median_entropy:.4f}): error rate = {high_error_rate:.1%}")
    print(f"Low uncertainty (<={median_entropy:.4f}): error rate = {low_error_rate:.1%}")
    
    analysis['median_entropy'] = median_entropy
    analysis['high_uncertainty_error_rate'] = high_error_rate
    analysis['low_uncertainty_error_rate'] = low_error_rate
    
    # Selective prediction analysis
    print("\n--- Selective Prediction (Reject Uncertain) ---")
    print(f"{'Coverage':>10} {'Accuracy':>10} {'Rejected':>10}")
    print("-" * 35)
    
    # Sort by uncertainty (low to high)
    sorted_results = sorted(results, key=lambda x: x['predictive_entropy'])
    
    for coverage in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        n_keep = int(len(sorted_results) * coverage)
        kept = sorted_results[:n_keep]
        
        if kept:
            accuracy = len([r for r in kept if r['result'] in ['TP', 'TN']]) / len(kept)
            print(f"{coverage:>9.0%} {accuracy:>10.1%} {1-coverage:>10.0%}")
            analysis[f'selective_accuracy_{int(coverage*100)}'] = accuracy
    
    return analysis

# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_results(results, analysis, dataset_name, timestamp):
    """Save results to CSV files."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not results:
        print("No results to save!")
        return
    
    # 1. Save all results with uncertainty
    all_file = os.path.join(OUTPUT_DIR, f"mc_dropout_{dataset_name}_{timestamp}.csv")
    fieldnames = list(results[0].keys())
    
    with open(all_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✓ All results: {all_file}")
    
    # 2. Save high-uncertainty samples (for review)
    median_entropy = np.median([r['predictive_entropy'] for r in results])
    high_uncertainty = [r for r in results if r['predictive_entropy'] > median_entropy]
    
    if high_uncertainty:
        high_unc_file = os.path.join(OUTPUT_DIR, f"high_uncertainty_{dataset_name}_{timestamp}.csv")
        with open(high_unc_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted(high_uncertainty, key=lambda x: -x['predictive_entropy']))
        print(f"✓ High uncertainty samples: {high_unc_file}")
    
    # 3. Save errors with uncertainty
    errors = [r for r in results if r['result'] in ['FP', 'FN']]
    if errors:
        errors_file = os.path.join(OUTPUT_DIR, f"errors_with_uncertainty_{dataset_name}_{timestamp}.csv")
        with open(errors_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted(errors, key=lambda x: -x['predictive_entropy']))
        print(f"✓ Errors with uncertainty: {errors_file}")
    
    # 4. Save analysis summary
    summary_file = os.path.join(OUTPUT_DIR, f"mc_dropout_summary_{dataset_name}_{timestamp}.csv")
    
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
    
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['=== MC DROPOUT ANALYSIS SUMMARY ===', ''])
        writer.writerow(['Dataset', dataset_name])
        writer.writerow(['Timestamp', timestamp])
        writer.writerow(['Model', MODEL_PATH])
        writer.writerow(['MC Passes', MC_DROPOUT_PASSES])
        writer.writerow(['Threshold', COUGH_THRESHOLD])
        writer.writerow(['Total Files', total])
        writer.writerow(['', ''])
        
        # Classification results
        writer.writerow(['=== CLASSIFICATION ===', ''])
        writer.writerow(['TP', tp])
        writer.writerow(['TN', tn])
        writer.writerow(['FP', fp])
        writer.writerow(['FN', fn])
        writer.writerow(['Accuracy', f"{accuracy:.4f}"])
        writer.writerow(['Precision', f"{precision:.4f}"])
        writer.writerow(['Recall', f"{recall:.4f}"])
        writer.writerow(['F1 Score', f"{f1:.4f}"])
        writer.writerow(['', ''])
        
        # Uncertainty analysis
        writer.writerow(['=== UNCERTAINTY ANALYSIS ===', ''])
        for key, value in analysis.items():
            if isinstance(value, float):
                writer.writerow([key, f"{value:.4f}"])
            else:
                writer.writerow([key, value])
    
    print(f"✓ Summary: {summary_file}")


def print_summary(results, dataset_name):
    """Print summary to console."""
    
    print("\n" + "=" * 70)
    print(f"MC DROPOUT RESULTS SUMMARY - {dataset_name}")
    print("=" * 70)
    
    total = len(results)
    tp = sum(1 for r in results if r['result'] == 'TP')
    tn = sum(1 for r in results if r['result'] == 'TN')
    fp = sum(1 for r in results if r['result'] == 'FP')
    fn = sum(1 for r in results if r['result'] == 'FN')
    
    print(f"\nTotal files: {total}")
    print(f"MC Dropout passes: {MC_DROPOUT_PASSES}")
    print(f"Threshold: {COUGH_THRESHOLD}")
    
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
    
    # Uncertainty stats
    print(f"\nUncertainty Statistics:")
    avg_entropy = np.mean([r['predictive_entropy'] for r in results])
    avg_epistemic = np.mean([r['epistemic_uncertainty'] for r in results])
    avg_aleatoric = np.mean([r['aleatoric_uncertainty'] for r in results])
    avg_std = np.mean([r['cough_prob_std'] for r in results])
    
    print(f"  Avg Predictive Entropy: {avg_entropy:.4f}")
    print(f"  Avg Epistemic (model):  {avg_epistemic:.4f}")
    print(f"  Avg Aleatoric (data):   {avg_aleatoric:.4f}")
    print(f"  Avg Cough Prob Std:     {avg_std:.4f}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run MC Dropout uncertainty quantification on datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_mc_dropout.py esc50
    python run_mc_dropout.py coswara
    python run_mc_dropout.py both
        """
    )
    parser.add_argument('dataset', choices=['coswara', 'esc50', 'both'],
                        help='Which dataset to process')
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("CoughGuard - Monte Carlo Dropout Uncertainty Quantification")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  MC Dropout passes: {MC_DROPOUT_PASSES}")
    print(f"  Dropout rate: {DROPOUT_RATE}")
    print(f"  Threshold: {COUGH_THRESHOLD}")
    print(f"  Dataset: {args.dataset.upper()}")
    print(f"  Timestamp: {timestamp}")
    
    # Check paths
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model not found: {MODEL_PATH}")
        return
    
    # Load model
    print("\n1. Loading model...")
    model = load_model()
    print("   ✓ Model loaded!")
    
    # Verify dropout exists
    dropout_count = sum(1 for m in model.modules() if isinstance(m, nn.Dropout))
    print(f"   ✓ Dropout layers found: {dropout_count}")
    
    # Process datasets
    if args.dataset in ['esc50', 'both']:
        print("\n2. Running MC Dropout inference on ESC-50...")
        esc50_results = process_esc50_with_mc_dropout(model)
        
        if esc50_results:
            print_summary(esc50_results, "ESC-50")
            analysis = analyze_uncertainty_vs_errors(esc50_results)
            save_results(esc50_results, analysis, "esc50", timestamp)
    
    if args.dataset in ['coswara', 'both']:
        print("\n3. Running MC Dropout inference on Coswara...")
        coswara_results = process_coswara_with_mc_dropout(model)
        
        if coswara_results:
            print_summary(coswara_results, "Coswara")
            analysis = analyze_uncertainty_vs_errors(coswara_results)
            save_results(coswara_results, analysis, "coswara", timestamp)
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
