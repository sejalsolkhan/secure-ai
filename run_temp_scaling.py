"""
CoughGuard - Temperature Scaling with Per-Sample Uncertainty Output
====================================================================

This script extends the temperature scaling pipeline to output per-sample data
for linking uncertainty to error types (ISEF analysis).

Outputs:
1. Summary statistics (same as before)
2. Per-sample CSV with ALL samples containing:
   - Metadata (filename, category, etc.)
   - MC Dropout stats (epistemic, aleatoric, std, etc.) - computed at T=1
   - Calibrated outputs (confidence, prediction, result) - after T scaling

Usage:
    python run_temp_scaling_with_per_sample.py esc50
    python run_temp_scaling_with_per_sample.py coswara
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
from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split
from models.mn.model import get_model
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

COSWARA_PATH = "datasets/Coswara-Data/Extracted_data"
ESC50_AUDIO_PATH = "datasets/ESC-50/audio"
ESC50_META_PATH = "datasets/ESC-50/meta/esc50.csv"

MODEL_PATH = "models/mn/mn10_mydata_epoch_13.pt"
OUTPUT_DIR = "outputs"

# Preprocessing parameters
SAMPLE_RATE = 32000
N_FFT = 1024
HOP_LENGTH = 320
N_MELS = 128
WIN_LENGTH = 800
FMIN = 0.0
FMAX = 15000

# Model configuration
NUM_CLASSES = 2
COUGH_CLASS_ID = 0

COSWARA_COUGH_FILES = ['cough-heavy.wav', 'cough-shallow.wav']
ESC50_COUGH_CATEGORY = 'coughing'

# Original clinical threshold (for comparison)
ORIGINAL_THRESHOLD = 0.7

# MC Dropout passes
MC_DROPOUT_PASSES = 20

# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess_audio(file_path: str):
    """Convert audio file to mel-spectrogram using lab's method."""
    
    waveform, sr = torchaudio.load(file_path)
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    if sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    
    preemphasis_coefficient = torch.tensor([[[-.97, 1]]])
    waveform = torch.nn.functional.conv1d(
        waveform.unsqueeze(0),
        preemphasis_coefficient
    ).squeeze(0)
    
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
        N_MELS, N_FFT, SAMPLE_RATE, FMIN, FMAX,
        vtln_low=100.0, vtln_high=-500., vtln_warp_factor=1.0
    )
    mel_basis = torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0)
    
    melspec = torch.matmul(mel_basis, power_spec.squeeze(0))
    melspec = (melspec + 0.00001).log()
    melspec = (melspec + 4.5) / 5.0
    
    melspec = melspec.unsqueeze(0).unsqueeze(0)
    
    return melspec

# =============================================================================
# MODEL
# =============================================================================

def load_model():
    """Load the lab's model."""
    
    model = get_model(
        num_classes=NUM_CLASSES,
        pretrained_name=None,
        width_mult=1.0,
        head_type="mlp",
        se_dims='c'
    )
    
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    model.load_state_dict(state_dict, strict=False)
    
    return model


def enable_mc_dropout(model):
    """Enable dropout during inference."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    return model

# =============================================================================
# MC DROPOUT UNCERTAINTY COMPUTATION
# =============================================================================

def compute_mc_dropout_stats(pass_logits):
    """
    Compute MC Dropout uncertainty statistics from multiple forward passes.
    
    Args:
        pass_logits: List of logits from MC Dropout passes, each shape (1, 2)
    
    Returns:
        Dictionary with uncertainty metrics
    """
    # Stack logits: shape (n_passes, 2)
    logits_array = np.array([p[0] for p in pass_logits])
    
    # Compute softmax for each pass
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    pass_probs = np.array([softmax(logits) for logits in logits_array])  # (n_passes, 2)
    
    # Cough probabilities across passes
    cough_probs = pass_probs[:, COUGH_CLASS_ID]  # (n_passes,)
    
    # Mean logits (for temperature scaling later)
    mean_logits = np.mean(logits_array, axis=0)  # (2,)
    
    # MC Dropout stats on raw probabilities
    cough_prob_mean_raw = np.mean(cough_probs)
    cough_prob_std = np.std(cough_probs)
    cough_prob_variance = np.var(cough_probs)
    cough_prob_min = np.min(cough_probs)
    cough_prob_max = np.max(cough_probs)
    
    # Mean prediction (probability)
    mean_probs = np.mean(pass_probs, axis=0)  # (2,)
    
    # Predictive entropy: H(mean prediction)
    mean_probs_clipped = np.clip(mean_probs, 1e-10, 1.0)
    predictive_entropy = -np.sum(mean_probs_clipped * np.log(mean_probs_clipped))
    
    # Aleatoric uncertainty: mean of entropy per pass
    def entropy(p):
        p_clipped = np.clip(p, 1e-10, 1.0)
        return -np.sum(p_clipped * np.log(p_clipped))
    
    per_pass_entropy = np.array([entropy(p) for p in pass_probs])
    aleatoric_uncertainty = np.mean(per_pass_entropy)
    
    # Epistemic uncertainty: predictive - aleatoric
    epistemic_uncertainty = predictive_entropy - aleatoric_uncertainty
    
    return {
        'mean_logits': mean_logits,
        'cough_prob_mean_raw': cough_prob_mean_raw,
        'cough_prob_std': cough_prob_std,
        'cough_prob_variance': cough_prob_variance,
        'cough_prob_min': cough_prob_min,
        'cough_prob_max': cough_prob_max,
        'predictive_entropy': predictive_entropy,
        'epistemic_uncertainty': epistemic_uncertainty,
        'aleatoric_uncertainty': aleatoric_uncertainty,
    }

# =============================================================================
# TEMPERATURE SCALING
# =============================================================================

def apply_temperature(logits, temperature):
    """Apply temperature scaling to logits and return probabilities."""
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probs


def find_optimal_temperature(logits, labels):
    """Find optimal temperature by minimizing NLL."""
    
    def nll_with_temperature(T):
        probs = apply_temperature(logits, T)
        true_class_probs = probs[np.arange(len(labels)), labels]
        true_class_probs = np.clip(true_class_probs, 1e-10, 1.0)
        return -np.mean(np.log(true_class_probs))
    
    result = minimize_scalar(nll_with_temperature, bounds=(0.1, 15.0), method='bounded')
    return result.x

# =============================================================================
# THRESHOLD OPTIMIZATION
# =============================================================================

def find_optimal_threshold(probs, labels, metric='f1'):
    """Find optimal threshold on validation data."""
    cough_probs = probs[:, 0]
    
    best_threshold = 0.5
    best_score = -1
    
    for threshold in np.arange(0.05, 0.95, 0.01):
        predictions = (cough_probs > threshold).astype(int)
        pred_labels = 1 - predictions
        
        tp = np.sum((pred_labels == 0) & (labels == 0))
        fn = np.sum((pred_labels == 1) & (labels == 0))
        fp = np.sum((pred_labels == 0) & (labels == 1))
        tn = np.sum((pred_labels == 1) & (labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        if metric == 'f1':
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        elif metric == 'youden':
            score = recall + specificity - 1
        elif metric == 'balanced_accuracy':
            score = (recall + specificity) / 2
        else:
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score

# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================

def compute_metrics(probs, labels, threshold):
    """Compute full classification metrics."""
    
    cough_probs = probs[:, 0]
    predictions = (cough_probs > threshold).astype(int)
    pred_labels = 1 - predictions
    
    tp = np.sum((pred_labels == 0) & (labels == 0))
    fn = np.sum((pred_labels == 1) & (labels == 0))
    fp = np.sum((pred_labels == 0) & (labels == 1))
    tn = np.sum((pred_labels == 1) & (labels == 1))
    
    total = len(labels)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'total': total,
        'predictions': pred_labels,
    }


def compute_calibration_metrics(probs, labels, predictions):
    """Compute ECE and confidence metrics."""
    
    confidences = np.where(predictions == 0, probs[:, 0], probs[:, 1])
    correct = predictions == labels
    
    # ECE
    n_bins = 15
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.sum(in_bin) > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(correct[in_bin])
            bin_weight = np.sum(in_bin) / len(confidences)
            ece += bin_weight * np.abs(avg_accuracy - avg_confidence)
    
    # Confidence gap
    conf_correct = np.mean(confidences[correct]) if np.sum(correct) > 0 else 0
    conf_incorrect = np.mean(confidences[~correct]) if np.sum(~correct) > 0 else 0
    
    # High/low confidence error rates
    median_conf = np.median(confidences)
    high_conf_mask = confidences > median_conf
    high_conf_error = np.mean(~correct[high_conf_mask]) if np.sum(high_conf_mask) > 0 else 0
    low_conf_error = np.mean(~correct[~high_conf_mask]) if np.sum(~high_conf_mask) > 0 else 0
    
    return {
        'ece': ece,
        'conf_gap': conf_correct - conf_incorrect,
        'high_conf_error_rate': high_conf_error,
        'low_conf_error_rate': low_conf_error,
        'avg_confidence': np.mean(confidences),
    }


def selective_prediction_analysis(probs, labels, predictions):
    """Analyze selective prediction with recall tracking."""
    
    confidences = np.where(predictions == 0, probs[:, 0], probs[:, 1])
    correct = predictions == labels
    
    sorted_indices = np.argsort(-confidences)
    
    results = []
    for coverage in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        n_keep = int(len(sorted_indices) * coverage)
        kept_indices = sorted_indices[:n_keep]
        
        if len(kept_indices) > 0:
            kept_correct = correct[kept_indices]
            kept_labels = labels[kept_indices]
            kept_preds = predictions[kept_indices]
            
            selective_accuracy = np.mean(kept_correct)
            
            kept_tp = np.sum((kept_preds == 0) & (kept_labels == 0))
            kept_actual_coughs = np.sum(kept_labels == 0)
            selective_recall = kept_tp / kept_actual_coughs if kept_actual_coughs > 0 else 0
            
            results.append({
                'coverage': coverage,
                'accuracy': selective_accuracy,
                'recall': selective_recall,
                'n_coughs_kept': kept_actual_coughs,
            })
    
    return results

# =============================================================================
# DATA COLLECTION WITH MC DROPOUT STATS
# =============================================================================

def collect_data_esc50(model):
    """Collect logits and MC Dropout stats from ESC-50 dataset."""
    
    print("   Collecting data from ESC-50...")
    
    all_data = []
    
    meta = pd.read_csv(ESC50_META_PATH)
    enable_mc_dropout(model)
    
    for idx, row in meta.iterrows():
        file_path = os.path.join(ESC50_AUDIO_PATH, row['filename'])
        
        if not os.path.exists(file_path):
            continue
        
        is_cough = row['category'] == ESC50_COUGH_CATEGORY
        label = 0 if is_cough else 1
        
        try:
            mel_spec = preprocess_audio(file_path)
            
            # MC Dropout: collect all passes
            pass_logits = []
            with torch.no_grad():
                for _ in range(MC_DROPOUT_PASSES):
                    logits, _ = model(mel_spec)
                    pass_logits.append(logits.numpy())
            
            # Compute MC Dropout stats
            mc_stats = compute_mc_dropout_stats(pass_logits)
            
            # Store sample data
            sample_data = {
                'filename': row['filename'],
                'file_path': file_path,
                'category': row['category'],
                'participant_id': '',  # ESC-50 doesn't have participants
                'date_folder': '',     # ESC-50 doesn't have date folders
                'is_cough': is_cough,
                'label': label,
                'mean_logits': mc_stats['mean_logits'],
                'cough_prob_mean_raw': mc_stats['cough_prob_mean_raw'],
                'cough_prob_std': mc_stats['cough_prob_std'],
                'cough_prob_variance': mc_stats['cough_prob_variance'],
                'cough_prob_min': mc_stats['cough_prob_min'],
                'cough_prob_max': mc_stats['cough_prob_max'],
                'predictive_entropy': mc_stats['predictive_entropy'],
                'epistemic_uncertainty': mc_stats['epistemic_uncertainty'],
                'aleatoric_uncertainty': mc_stats['aleatoric_uncertainty'],
            }
            all_data.append(sample_data)
            
        except Exception as e:
            print(f"      Error: {row['filename']}: {e}")
        
        if (idx + 1) % 500 == 0:
            print(f"      Processed {idx + 1}/{len(meta)}")
    
    print(f"      Total: {len(all_data)} files")
    return all_data


def collect_data_coswara(model):
    """Collect logits and MC Dropout stats from Coswara dataset."""
    
    print("   Collecting data from Coswara...")
    
    all_data = []
    
    enable_mc_dropout(model)
    
    if not os.path.exists(COSWARA_PATH):
        print(f"❌ ERROR: Coswara path not found: {COSWARA_PATH}")
        return []

    date_folders = [d for d in os.listdir(COSWARA_PATH) 
                    if os.path.isdir(os.path.join(COSWARA_PATH, d))]
    
    print(f"   Found {len(date_folders)} date folders to process.")
    
    file_count = 0
    start_time = datetime.now()
    
    for i, date_folder in enumerate(sorted(date_folders)):
        date_path = os.path.join(COSWARA_PATH, date_folder)
        participants = [p for p in os.listdir(date_path)
                       if os.path.isdir(os.path.join(date_path, p))]
        
        for participant_id in participants:
            participant_path = os.path.join(date_path, participant_id)
            wav_files = [f for f in os.listdir(participant_path) if f.endswith('.wav')]
            
            for wav_file in wav_files:
                file_path = os.path.join(participant_path, wav_file)
                is_cough = wav_file in COSWARA_COUGH_FILES
                label = 0 if is_cough else 1
                category = wav_file.replace('.wav', '')
                
                try:
                    mel_spec = preprocess_audio(file_path)
                    
                    # MC Dropout: collect all passes
                    pass_logits = []
                    with torch.no_grad():
                        for _ in range(MC_DROPOUT_PASSES):
                            logits, _ = model(mel_spec)
                            pass_logits.append(logits.numpy())
                    
                    # Compute MC Dropout stats
                    mc_stats = compute_mc_dropout_stats(pass_logits)
                    
                    # Store sample data
                    sample_data = {
                        'filename': wav_file,
                        'file_path': file_path,
                        'category': category,
                        'participant_id': participant_id,
                        'date_folder': date_folder,
                        'is_cough': is_cough,
                        'label': label,
                        'mean_logits': mc_stats['mean_logits'],
                        'cough_prob_mean_raw': mc_stats['cough_prob_mean_raw'],
                        'cough_prob_std': mc_stats['cough_prob_std'],
                        'cough_prob_variance': mc_stats['cough_prob_variance'],
                        'cough_prob_min': mc_stats['cough_prob_min'],
                        'cough_prob_max': mc_stats['cough_prob_max'],
                        'predictive_entropy': mc_stats['predictive_entropy'],
                        'epistemic_uncertainty': mc_stats['epistemic_uncertainty'],
                        'aleatoric_uncertainty': mc_stats['aleatoric_uncertainty'],
                    }
                    all_data.append(sample_data)
                    
                    file_count += 1
                    
                    if file_count % 500 == 0:
                        elapsed = datetime.now() - start_time
                        print(f"      ➤ Processed {file_count} files... (Time: {elapsed})", flush=True)
                    
                except Exception as e:
                    pass
    
    print(f"      Total: {file_count} files")
    return all_data

# =============================================================================
# PER-SAMPLE OUTPUT
# =============================================================================

def save_per_sample_results(all_data, optimal_temp, optimal_threshold, dataset_name, timestamp):
    """
    Apply temperature scaling and threshold to all samples, then save per-sample CSV.
    """
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Build arrays for vectorized operations
    logits = np.array([d['mean_logits'] for d in all_data])
    labels = np.array([d['label'] for d in all_data])
    
    # Apply temperature scaling
    probs_calibrated = apply_temperature(logits, optimal_temp)
    cough_probs_calibrated = probs_calibrated[:, COUGH_CLASS_ID]
    
    # Apply threshold
    predictions = (cough_probs_calibrated > optimal_threshold).astype(int)
    pred_labels = 1 - predictions  # 0=cough, 1=not_cough
    
    # Compute calibrated confidence (confidence in the predicted class)
    confidence_calibrated = np.where(pred_labels == 0, 
                                      probs_calibrated[:, 0], 
                                      probs_calibrated[:, 1])
    
    # Determine result (TP/TN/FP/FN)
    results = []
    for i in range(len(labels)):
        pred = pred_labels[i]
        true = labels[i]
        if pred == 0 and true == 0:
            results.append('TP')
        elif pred == 1 and true == 1:
            results.append('TN')
        elif pred == 0 and true == 1:
            results.append('FP')
        else:  # pred == 1 and true == 0
            results.append('FN')
    
    # Build output DataFrame
    output_rows = []
    for i, d in enumerate(all_data):
        row = {
            # Metadata
            'filename': d['filename'],
            'file_path': d['file_path'],
            'category': d['category'],
            'participant_id': d['participant_id'],
            'date_folder': d['date_folder'],
            'is_cough': d['is_cough'],
            
            # MC Dropout stats (T=1)
            'cough_prob_mean_raw': d['cough_prob_mean_raw'],
            'cough_prob_std': d['cough_prob_std'],
            'cough_prob_variance': d['cough_prob_variance'],
            'cough_prob_min': d['cough_prob_min'],
            'cough_prob_max': d['cough_prob_max'],
            'predictive_entropy': d['predictive_entropy'],
            'epistemic_uncertainty': d['epistemic_uncertainty'],
            'aleatoric_uncertainty': d['aleatoric_uncertainty'],
            
            # Calibrated outputs (after T scaling + threshold)
            'cough_prob_calibrated': cough_probs_calibrated[i],
            'confidence_calibrated': confidence_calibrated[i],
            'model_prediction': 'cough' if pred_labels[i] == 0 else 'not_cough',
            'result': results[i],
        }
        output_rows.append(row)
    
    # Save to CSV
    df = pd.DataFrame(output_rows)
    
    # Format temperature and threshold for filename
    temp_str = f"{optimal_temp:.2f}".replace('.', 'p')
    thresh_str = f"{optimal_threshold:.2f}".replace('.', 'p')
    
    output_file = os.path.join(OUTPUT_DIR, f"all_samples_T{temp_str}_th{thresh_str}_{dataset_name}_{timestamp}.csv")
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Per-sample results saved: {output_file}")
    print(f"  Total samples: {len(df)}")
    print(f"  TP: {len(df[df['result']=='TP'])}, TN: {len(df[df['result']=='TN'])}, FP: {len(df[df['result']=='FP'])}, FN: {len(df[df['result']=='FN'])}")
    
    return df

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_analysis(all_data, dataset_name, timestamp):
    """Run temperature scaling with threshold recalibration and save per-sample results."""
    
    print("\n" + "=" * 70)
    print("TEMPERATURE SCALING WITH THRESHOLD RECALIBRATION")
    print("=" * 70)
    
    # Extract arrays
    logits = np.array([d['mean_logits'] for d in all_data])
    labels = np.array([d['label'] for d in all_data])
    
    # =========================================================================
    # Split data
    # =========================================================================
    print("\n1. Splitting data into Validation (50%) and Test (50%) sets...")
    
    indices = np.arange(len(logits))
    val_idx, test_idx = train_test_split(indices, test_size=0.5, random_state=42, stratify=labels)
    
    val_logits, test_logits = logits[val_idx], logits[test_idx]
    val_labels, test_labels = labels[val_idx], labels[test_idx]
    
    print(f"   Validation: {len(val_idx)} samples ({np.sum(val_labels == 0)} coughs)")
    print(f"   Test: {len(test_idx)} samples ({np.sum(test_labels == 0)} coughs)")
    
    # =========================================================================
    # Find optimal temperature on validation
    # =========================================================================
    print("\n2. Finding optimal temperature on validation set...")
    optimal_temp = find_optimal_temperature(val_logits, val_labels)
    print(f"   ✓ Optimal temperature: {optimal_temp:.4f}")
    
    # =========================================================================
    # Find optimal threshold on validation (AFTER temperature scaling)
    # =========================================================================
    print("\n3. Finding optimal threshold on validation set (after T scaling)...")
    val_probs_scaled = apply_temperature(val_logits, optimal_temp)
    optimal_threshold, best_f1 = find_optimal_threshold(val_probs_scaled, val_labels, metric='f1')
    print(f"   ✓ Optimal threshold: {optimal_threshold:.4f} (F1={best_f1:.4f})")
    
    # =========================================================================
    # Evaluate on TEST set
    # =========================================================================
    print("\n4. Evaluating on held-out test set...")
    
    # Method 1: Original (T=1, threshold=0.7)
    probs_original = apply_temperature(test_logits, 1.0)
    metrics_original = compute_metrics(probs_original, test_labels, ORIGINAL_THRESHOLD)
    calib_original = compute_calibration_metrics(probs_original, test_labels, metrics_original['predictions'])
    select_original = selective_prediction_analysis(probs_original, test_labels, metrics_original['predictions'])
    
    # Method 2: Temp Scaling + Recalibrated Threshold
    probs_scaled = apply_temperature(test_logits, optimal_temp)
    metrics_scaled = compute_metrics(probs_scaled, test_labels, optimal_threshold)
    calib_scaled = compute_calibration_metrics(probs_scaled, test_labels, metrics_scaled['predictions'])
    select_scaled = selective_prediction_analysis(probs_scaled, test_labels, metrics_scaled['predictions'])
    
    # =========================================================================
    # Print Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS ON HELD-OUT TEST SET")
    print("=" * 70)
    
    print(f"\n--- Configuration ---")
    print(f"Original: T=1.0, threshold={ORIGINAL_THRESHOLD}")
    print(f"Recalibrated: T={optimal_temp:.4f}, threshold={optimal_threshold:.4f}")
    
    print(f"\n--- Classification Metrics ---")
    print(f"{'Metric':<15} {'Original':>15} {'Recalibrated':>15} {'Change':>12}")
    print("-" * 60)
    
    for metric in ['accuracy', 'precision', 'recall', 'specificity', 'f1']:
        o = metrics_original[metric]
        r = metrics_scaled[metric]
        change = (r - o) * 100
        flag = ""
        if metric == 'recall' and r < o:
            flag = "⚠️"
        elif metric == 'recall' and r > o:
            flag = "✓"
        print(f"{metric.capitalize():<15} {o:>14.1%} {r:>15.1%} {change:>+11.1f}% {flag}")
    
    print(f"\n--- Confusion Matrix (Original: T=1.0, threshold={ORIGINAL_THRESHOLD}) ---")
    print(f"                    Predicted")
    print(f"                 Cough    Not-Cough")
    print(f"  Actual Cough    {metrics_original['tp']:4d}       {metrics_original['fn']:4d}")
    print(f"  Actual Other    {metrics_original['fp']:4d}       {metrics_original['tn']:4d}")
    
    print(f"\n--- Confusion Matrix (Recalibrated: T={optimal_temp:.2f}, threshold={optimal_threshold:.2f}) ---")
    print(f"                    Predicted")
    print(f"                 Cough    Not-Cough")
    print(f"  Actual Cough    {metrics_scaled['tp']:4d}       {metrics_scaled['fn']:4d}")
    print(f"  Actual Other    {metrics_scaled['fp']:4d}       {metrics_scaled['tn']:4d}")
    
    print(f"\n--- Calibration & Uncertainty ---")
    print(f"{'Metric':<25} {'Original':>12} {'Recalibrated':>12} {'Better?':>10}")
    print("-" * 62)
    
    print(f"{'ECE':<25} {calib_original['ece']:>12.4f} {calib_scaled['ece']:>12.4f} {'✓' if calib_scaled['ece'] < calib_original['ece'] else '✗':>10}")
    print(f"{'Confidence Gap':<25} {calib_original['conf_gap']:>12.4f} {calib_scaled['conf_gap']:>12.4f} {'✓' if calib_scaled['conf_gap'] > calib_original['conf_gap'] else '✗':>10}")
    print(f"{'High Conf Error Rate':<25} {calib_original['high_conf_error_rate']:>11.1%} {calib_scaled['high_conf_error_rate']:>12.1%} {'✓' if calib_scaled['high_conf_error_rate'] < calib_original['high_conf_error_rate'] else '✗':>10}")
    
    print(f"\n--- Selective Prediction ---")
    print(f"{'Coverage':<10} {'Acc (Orig)':>12} {'Acc (Recal)':>12} {'Rec (Orig)':>12} {'Rec (Recal)':>12}")
    print("-" * 62)
    
    for so, sr in zip(select_original, select_scaled):
        print(f"{so['coverage']:>8.0%} {so['accuracy']:>12.1%} {sr['accuracy']:>12.1%} {so['recall']:>12.1%} {sr['recall']:>12.1%}")
    
    # =========================================================================
    # Summary Assessment
    # =========================================================================
    print(f"\n" + "=" * 70)
    print("SUMMARY ASSESSMENT")
    print("=" * 70)
    
    recall_improved = metrics_scaled['recall'] > metrics_original['recall']
    precision_improved = metrics_scaled['precision'] > metrics_original['precision']
    f1_improved = metrics_scaled['f1'] > metrics_original['f1']
    conf_gap_improved = calib_scaled['conf_gap'] > calib_original['conf_gap']
    
    print(f"\nRecall:      {'✓ Improved' if recall_improved else '✗ Decreased'} ({metrics_original['recall']:.1%} → {metrics_scaled['recall']:.1%})")
    print(f"Precision:   {'✓ Improved' if precision_improved else '✗ Decreased'} ({metrics_original['precision']:.1%} → {metrics_scaled['precision']:.1%})")
    print(f"F1 Score:    {'✓ Improved' if f1_improved else '✗ Decreased'} ({metrics_original['f1']:.1%} → {metrics_scaled['f1']:.1%})")
    print(f"Conf Gap:    {'✓ Improved' if conf_gap_improved else '✗ Decreased'} ({calib_original['conf_gap']:.4f} → {calib_scaled['conf_gap']:.4f})")
    
    if recall_improved and conf_gap_improved:
        print(f"\n✓ SUCCESS: Both recall and calibration improved!")
    elif recall_improved and not conf_gap_improved:
        print(f"\n⚠️ PARTIAL: Recall improved but calibration did not.")
    elif not recall_improved and conf_gap_improved:
        print(f"\n⚠️ PARTIAL: Calibration improved but recall decreased.")
    else:
        print(f"\n✗ Neither recall nor calibration improved significantly.")
    
    # =========================================================================
    # Save Summary Results
    # =========================================================================
    save_summary_results(
        optimal_temp, optimal_threshold,
        metrics_original, metrics_scaled,
        calib_original, calib_scaled,
        select_original, select_scaled,
        dataset_name, timestamp
    )
    
    # =========================================================================
    # Save Per-Sample Results (ALL samples, not just test set)
    # =========================================================================
    print(f"\n" + "=" * 70)
    print("SAVING PER-SAMPLE RESULTS (ALL SAMPLES)")
    print("=" * 70)
    
    save_per_sample_results(all_data, optimal_temp, optimal_threshold, dataset_name, timestamp)
    
    return optimal_temp, optimal_threshold


def save_summary_results(optimal_temp, optimal_threshold, metrics_orig, metrics_scaled,
                         calib_orig, calib_scaled, select_orig, select_scaled,
                         dataset_name, timestamp):
    """Save summary results to CSV."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    summary_file = os.path.join(OUTPUT_DIR, f"summary_temp_scaling_{dataset_name}_{timestamp}.csv")
    
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow(['=== TEMPERATURE SCALING + THRESHOLD RECALIBRATION ===', ''])
        writer.writerow(['Dataset', dataset_name])
        writer.writerow(['Timestamp', timestamp])
        writer.writerow(['', ''])
        
        writer.writerow(['=== OPTIMIZED PARAMETERS ===', ''])
        writer.writerow(['Optimal Temperature', f"{optimal_temp:.4f}"])
        writer.writerow(['Optimal Threshold', f"{optimal_threshold:.4f}"])
        writer.writerow(['Original Threshold', ORIGINAL_THRESHOLD])
        writer.writerow(['', ''])
        
        writer.writerow(['=== CLASSIFICATION (Test Set) ===', '', ''])
        writer.writerow(['Metric', 'Original', 'Recalibrated'])
        writer.writerow(['Accuracy', f"{metrics_orig['accuracy']:.4f}", f"{metrics_scaled['accuracy']:.4f}"])
        writer.writerow(['Precision', f"{metrics_orig['precision']:.4f}", f"{metrics_scaled['precision']:.4f}"])
        writer.writerow(['RECALL', f"{metrics_orig['recall']:.4f}", f"{metrics_scaled['recall']:.4f}"])
        writer.writerow(['Specificity', f"{metrics_orig['specificity']:.4f}", f"{metrics_scaled['specificity']:.4f}"])
        writer.writerow(['F1', f"{metrics_orig['f1']:.4f}", f"{metrics_scaled['f1']:.4f}"])
        writer.writerow(['', ''])
        
        writer.writerow(['=== CONFUSION MATRIX (Original) ===', ''])
        writer.writerow(['TP', metrics_orig['tp']])
        writer.writerow(['FN', metrics_orig['fn']])
        writer.writerow(['FP', metrics_orig['fp']])
        writer.writerow(['TN', metrics_orig['tn']])
        writer.writerow(['', ''])
        
        writer.writerow(['=== CONFUSION MATRIX (Recalibrated) ===', ''])
        writer.writerow(['TP', metrics_scaled['tp']])
        writer.writerow(['FN', metrics_scaled['fn']])
        writer.writerow(['FP', metrics_scaled['fp']])
        writer.writerow(['TN', metrics_scaled['tn']])
        writer.writerow(['', ''])
        
        writer.writerow(['=== CALIBRATION ===', '', ''])
        writer.writerow(['Metric', 'Original', 'Recalibrated'])
        writer.writerow(['ECE', f"{calib_orig['ece']:.4f}", f"{calib_scaled['ece']:.4f}"])
        writer.writerow(['Confidence Gap', f"{calib_orig['conf_gap']:.4f}", f"{calib_scaled['conf_gap']:.4f}"])
        writer.writerow(['High Conf Error Rate', f"{calib_orig['high_conf_error_rate']:.4f}", f"{calib_scaled['high_conf_error_rate']:.4f}"])
    
    print(f"\n✓ Summary results saved: {summary_file}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Temperature scaling with per-sample output')
    parser.add_argument('dataset', choices=['esc50', 'coswara'],
                        help='Which dataset to process')
    
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("CoughGuard - Temperature Scaling + Per-Sample Uncertainty Output")
    print("=" * 70)
    print(f"\nDataset: {args.dataset.upper()}")
    print(f"Original threshold: {ORIGINAL_THRESHOLD}")
    print(f"\nMethod:")
    print(f"  1. Collect data with MC Dropout stats")
    print(f"  2. Find optimal T on validation set")
    print(f"  3. Find optimal threshold on validation set (after T applied)")
    print(f"  4. Evaluate on held-out test set")
    print(f"  5. Save per-sample results for ALL samples")
    
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model not found: {MODEL_PATH}")
        return
    
    print("\n" + "-" * 70)
    print("Loading model...")
    model = load_model()
    print("✓ Model loaded!")
    
    print("\nCollecting data (with MC Dropout stats)...")
    if args.dataset == 'esc50':
        all_data = collect_data_esc50(model)
    else:
        all_data = collect_data_coswara(model)
    
    print(f"✓ Collected {len(all_data)} samples")
    cough_count = sum(1 for d in all_data if d['is_cough'])
    print(f"✓ Cough samples: {cough_count} ({100*cough_count/len(all_data):.1f}%)")
    
    run_analysis(all_data, args.dataset, timestamp)
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
