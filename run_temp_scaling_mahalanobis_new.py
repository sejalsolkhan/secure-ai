"""
CoughGuard - Temperature Scaling with Mahalanobis Distance (CORRECTED)
=======================================================================

CORRECTED VERSION: Proper train/validation/test methodology

Key fixes:
1. Mahalanobis params computed from VALIDATION TPs only (not test)
2. Split column added to output CSV
3. All statistics computed on held-out test set

Methodology:
- 50/50 Validation/Test split (stratified)
- Validation set: Optimize T, threshold, AND Mahalanobis params
- Test set: Final evaluation only (never touched during fitting)

Usage:
    python run_temp_scaling_mahalanobis_new.py esc50
    python run_temp_scaling_mahalanobis_new.py coswara
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
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from scipy.spatial.distance import mahalanobis
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

# Mahalanobis threshold candidates for tuning
MAHALANOBIS_THRESHOLDS = [30, 35, 40, 45, 50]

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
    """Compute MC Dropout uncertainty statistics from multiple forward passes."""
    
    logits_array = np.array([p[0] for p in pass_logits])
    
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    pass_probs = np.array([softmax(logits) for logits in logits_array])
    cough_probs = pass_probs[:, COUGH_CLASS_ID]
    mean_logits = np.mean(logits_array, axis=0)
    
    cough_prob_mean_raw = np.mean(cough_probs)
    cough_prob_std = np.std(cough_probs)
    cough_prob_variance = np.var(cough_probs)
    cough_prob_min = np.min(cough_probs)
    cough_prob_max = np.max(cough_probs)
    
    mean_probs = np.mean(pass_probs, axis=0)
    predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
    
    expected_entropy = np.mean([
        -np.sum(p * np.log(p + 1e-10)) for p in pass_probs
    ])
    
    epistemic_uncertainty = predictive_entropy - expected_entropy
    aleatoric_uncertainty = expected_entropy
    
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
    """Apply temperature scaling to logits."""
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return probs


def find_optimal_temperature(logits, labels, min_temp=0.1, max_temp=15.0):
    """Find optimal temperature using NLL on validation set."""
    
    def nll_with_temperature(temperature):
        probs = apply_temperature(logits, temperature)
        cough_probs = probs[:, COUGH_CLASS_ID]
        
        label_probs = np.where(labels == 0, cough_probs, 1 - cough_probs)
        label_probs = np.clip(label_probs, 1e-10, 1 - 1e-10)
        
        nll = -np.mean(np.log(label_probs))
        return nll
    
    result = minimize_scalar(nll_with_temperature, bounds=(min_temp, max_temp), method='bounded')
    return result.x


def find_optimal_threshold(probs, labels, metric='f1'):
    """Find optimal classification threshold."""
    
    cough_probs = probs[:, COUGH_CLASS_ID]
    best_threshold = 0.5
    best_score = 0
    
    for threshold in np.arange(0.3, 0.8, 0.01):
        predictions = (cough_probs > threshold).astype(int)
        pred_labels = 1 - predictions
        
        tp = np.sum((pred_labels == 0) & (labels == 0))
        fp = np.sum((pred_labels == 0) & (labels == 1))
        fn = np.sum((pred_labels == 1) & (labels == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_score:
            best_score = f1
            best_threshold = threshold
    
    return best_threshold, best_score

# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_metrics(probs, labels, threshold):
    """Compute classification metrics."""
    
    cough_probs = probs[:, COUGH_CLASS_ID]
    predictions = (cough_probs > threshold).astype(int)
    pred_labels = 1 - predictions
    
    tp = np.sum((pred_labels == 0) & (labels == 0))
    tn = np.sum((pred_labels == 1) & (labels == 1))
    fp = np.sum((pred_labels == 0) & (labels == 1))
    fn = np.sum((pred_labels == 1) & (labels == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
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
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'predictions': pred_labels,
    }


def compute_calibration_metrics(probs, labels, predictions):
    """Compute calibration metrics (ECE, confidence gap)."""
    
    cough_probs = probs[:, COUGH_CLASS_ID]
    confidence = np.where(predictions == 0, cough_probs, 1 - cough_probs)
    correct = (predictions == labels).astype(float)
    
    # ECE
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    
    for i in range(n_bins):
        mask = (confidence >= bin_boundaries[i]) & (confidence < bin_boundaries[i + 1])
        if np.sum(mask) > 0:
            bin_accuracy = np.mean(correct[mask])
            bin_confidence = np.mean(confidence[mask])
            bin_size = np.sum(mask) / len(confidence)
            ece += bin_size * np.abs(bin_accuracy - bin_confidence)
    
    # Confidence gap
    correct_conf = np.mean(confidence[correct == 1]) if np.sum(correct) > 0 else 0
    incorrect_conf = np.mean(confidence[correct == 0]) if np.sum(1 - correct) > 0 else 0
    conf_gap = correct_conf - incorrect_conf
    
    return {
        'ece': ece,
        'conf_gap': conf_gap,
    }

# =============================================================================
# DATA COLLECTION (ESC-50)
# =============================================================================

def collect_data_esc50(model):
    """Collect data from ESC-50 dataset."""
    
    print("\n   Loading ESC-50 metadata...")
    meta_df = pd.read_csv(ESC50_META_PATH)
    
    all_data = []
    model = enable_mc_dropout(model)
    
    print(f"   Processing {len(meta_df)} files...")
    start_time = datetime.now()
    
    for idx, row in meta_df.iterrows():
        file_path = os.path.join(ESC50_AUDIO_PATH, row['filename'])
        
        if not os.path.exists(file_path):
            continue
        
        try:
            input_tensor = preprocess_audio(file_path)
            
            pass_logits = []
            pass_embeddings = []
            
            with torch.no_grad():
                for _ in range(MC_DROPOUT_PASSES):
                    logits, embedding = model(input_tensor)
                    pass_logits.append(logits.numpy())
                    pass_embeddings.append(embedding.numpy().flatten())
            
            mc_stats = compute_mc_dropout_stats(pass_logits)
            mean_embedding = np.mean(pass_embeddings, axis=0)
            
            is_cough = row['category'] == ESC50_COUGH_CATEGORY
            
            sample_data = {
                'filename': row['filename'],
                'file_path': file_path,
                'category': row['category'],
                'participant_id': row.get('fold', 'unknown'),
                'date_folder': '',
                'is_cough': is_cough,
                'label': 0 if is_cough else 1,
                'embedding': mean_embedding,
                **mc_stats,
            }
            
            all_data.append(sample_data)
            
            if (idx + 1) % 200 == 0:
                elapsed = datetime.now() - start_time
                print(f"      ➤ Processed {idx + 1}/{len(meta_df)} files... (Time: {elapsed})")
                
        except Exception as e:
            pass
    
    print(f"      Total: {len(all_data)} files")
    return all_data

# =============================================================================
# DATA COLLECTION (COSWARA)
# =============================================================================

def collect_data_coswara(model):
    """Collect data from Coswara dataset."""
    
    print(f"\n   Scanning Coswara directory: {COSWARA_PATH}")
    
    all_data = []
    model = enable_mc_dropout(model)
    
    date_folders = sorted([d for d in os.listdir(COSWARA_PATH) 
                          if os.path.isdir(os.path.join(COSWARA_PATH, d))])
    
    print(f"   Found {len(date_folders)} date folders")
    
    file_count = 0
    start_time = datetime.now()
    
    for date_folder in date_folders:
        date_path = os.path.join(COSWARA_PATH, date_folder)
        participant_folders = [d for d in os.listdir(date_path) 
                              if os.path.isdir(os.path.join(date_path, d))]
        
        for participant_id in participant_folders:
            participant_path = os.path.join(date_path, participant_id)
            
            audio_files = [f for f in os.listdir(participant_path) if f.endswith('.wav')]
            
            for audio_file in audio_files:
                file_path = os.path.join(participant_path, audio_file)
                
                try:
                    input_tensor = preprocess_audio(file_path)
                    
                    pass_logits = []
                    pass_embeddings = []
                    
                    with torch.no_grad():
                        for _ in range(MC_DROPOUT_PASSES):
                            logits, embedding = model(input_tensor)
                            pass_logits.append(logits.numpy())
                            pass_embeddings.append(embedding.numpy().flatten())
                    
                    mc_stats = compute_mc_dropout_stats(pass_logits)
                    mean_embedding = np.mean(pass_embeddings, axis=0)
                    
                    category = audio_file.replace('.wav', '')
                    is_cough = audio_file in COSWARA_COUGH_FILES
                    
                    sample_data = {
                        'filename': audio_file,
                        'file_path': file_path,
                        'category': category,
                        'participant_id': participant_id,
                        'date_folder': date_folder,
                        'is_cough': is_cough,
                        'label': 0 if is_cough else 1,
                        'embedding': mean_embedding,
                        **mc_stats,
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
# MAHALANOBIS DISTANCE COMPUTATION (CORRECTED)
# =============================================================================

def compute_mahalanobis_params_from_validation(all_data, val_idx, optimal_temp, optimal_threshold):
    """
    Compute Mahalanobis parameters from VALIDATION TPs only.
    
    CRITICAL: This ensures no data leakage - test set is not used for fitting.
    """
    
    print("\n" + "=" * 70)
    print("MAHALANOBIS PARAMETER FITTING (VALIDATION SET ONLY)")
    print("=" * 70)
    
    # Get validation data only
    val_data = [all_data[i] for i in val_idx]
    
    # Build arrays for validation set
    logits = np.array([d['mean_logits'] for d in val_data])
    labels = np.array([d['label'] for d in val_data])
    embeddings = np.array([d['embedding'] for d in val_data])
    
    print(f"\n   Validation set size: {len(val_data)}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    
    # Apply temperature scaling to determine TPs on validation set
    probs_calibrated = apply_temperature(logits, optimal_temp)
    cough_probs_calibrated = probs_calibrated[:, COUGH_CLASS_ID]
    predictions = (cough_probs_calibrated > optimal_threshold).astype(int)
    pred_labels = 1 - predictions
    
    # Identify VALIDATION TPs only
    is_tp = (pred_labels == 0) & (labels == 0)
    val_tp_embeddings = embeddings[is_tp]
    
    print(f"   Validation TPs: {len(val_tp_embeddings)}")
    
    if len(val_tp_embeddings) < 10:
        print(f"   ⚠️ Warning: Very few validation TPs!")
        return None, None
    
    # Fit covariance on VALIDATION TPs only
    if len(val_tp_embeddings) < embeddings.shape[1]:
        print(f"   Using Ledoit-Wolf regularization (samples < dimensions)")
        cov = LedoitWolf().fit(val_tp_embeddings)
    else:
        print(f"   Using Empirical Covariance")
        cov = EmpiricalCovariance().fit(val_tp_embeddings)
    
    tp_mean = cov.location_
    tp_precision = cov.precision_
    
    print(f"   ✓ Mahalanobis params fitted on {len(val_tp_embeddings)} validation TPs")
    
    return tp_mean, tp_precision


def compute_mahalanobis_distances_for_all(all_data, tp_mean, tp_precision):
    """
    Compute Mahalanobis distance for ALL samples using pre-fitted params.
    
    The params were fitted on validation TPs only, so this is safe.
    """
    
    print("\n   Computing Mahalanobis distances for all samples...")
    
    embeddings = np.array([d['embedding'] for d in all_data])
    
    distances = []
    for i, emb in enumerate(embeddings):
        try:
            d = mahalanobis(emb, tp_mean, tp_precision)
            distances.append(d)
        except Exception as e:
            d = np.linalg.norm(emb - tp_mean)
            distances.append(d)
        
        if (i + 1) % 5000 == 0:
            print(f"      Computed {i + 1}/{len(all_data)} distances...")
    
    # Add distances to all_data
    for i, d in enumerate(all_data):
        d['mahalanobis_distance'] = distances[i]
    
    return all_data


def find_optimal_mahalanobis_threshold(all_data, val_idx, optimal_temp, optimal_threshold):
    """
    Find optimal Mahalanobis threshold on VALIDATION set.
    
    Optimizes for best F1 on validation set after Mahalanobis filtering.
    """
    
    print("\n   Finding optimal Mahalanobis threshold (validation set)...")
    
    val_data = [all_data[i] for i in val_idx]
    
    logits = np.array([d['mean_logits'] for d in val_data])
    labels = np.array([d['label'] for d in val_data])
    distances = np.array([d['mahalanobis_distance'] for d in val_data])
    
    probs_calibrated = apply_temperature(logits, optimal_temp)
    cough_probs_calibrated = probs_calibrated[:, COUGH_CLASS_ID]
    predictions = (cough_probs_calibrated > optimal_threshold).astype(int)
    pred_labels = 1 - predictions
    
    best_threshold = 40.0
    best_f1 = 0
    
    for thresh in MAHALANOBIS_THRESHOLDS:
        # After Mahalanobis: reject "cough" predictions with distance > threshold
        final_pred = pred_labels.copy()
        final_pred[(pred_labels == 0) & (distances > thresh)] = 1  # Reject to "not cough"
        
        tp = np.sum((final_pred == 0) & (labels == 0))
        fp = np.sum((final_pred == 0) & (labels == 1))
        fn = np.sum((final_pred == 1) & (labels == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"      Threshold {thresh}: Precision={precision:.1%}, Recall={recall:.1%}, F1={f1:.1%}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    print(f"   ✓ Optimal Mahalanobis threshold: {best_threshold} (F1={best_f1:.1%})")
    
    return best_threshold

# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_per_sample_results(all_data, val_idx, test_idx, optimal_temp, optimal_threshold, 
                            mahal_threshold, dataset_name, timestamp):
    """Save per-sample results with split information."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create split labels
    split_labels = ['unknown'] * len(all_data)
    for i in val_idx:
        split_labels[i] = 'validation'
    for i in test_idx:
        split_labels[i] = 'test'
    
    # Build arrays for all samples
    logits = np.array([d['mean_logits'] for d in all_data])
    labels = np.array([d['label'] for d in all_data])
    
    probs_calibrated = apply_temperature(logits, optimal_temp)
    cough_probs_calibrated = probs_calibrated[:, COUGH_CLASS_ID]
    predictions = (cough_probs_calibrated > optimal_threshold).astype(int)
    pred_labels = 1 - predictions
    
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
        else:
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
            
            # CRITICAL: Split information
            'split': split_labels[i],
            
            # MC Dropout stats (T=1)
            'cough_prob_mean_raw': d['cough_prob_mean_raw'],
            'cough_prob_std': d['cough_prob_std'],
            'cough_prob_variance': d['cough_prob_variance'],
            'cough_prob_min': d['cough_prob_min'],
            'cough_prob_max': d['cough_prob_max'],
            'predictive_entropy': d['predictive_entropy'],
            'epistemic_uncertainty': d['epistemic_uncertainty'],
            'aleatoric_uncertainty': d['aleatoric_uncertainty'],
            
            # Calibrated outputs
            'cough_prob_calibrated': cough_probs_calibrated[i],
            'confidence_calibrated': confidence_calibrated[i],
            'model_prediction': 'cough' if pred_labels[i] == 0 else 'not_cough',
            'result': results[i],
            
            # Mahalanobis distance
            'mahalanobis_distance': d.get('mahalanobis_distance', np.nan),
        }
        output_rows.append(row)
    
    df = pd.DataFrame(output_rows)
    
    # Format filename
    temp_str = f"{optimal_temp:.2f}".replace('.', 'p')
    thresh_str = f"{optimal_threshold:.2f}".replace('.', 'p')
    
    output_file = os.path.join(OUTPUT_DIR, f"all_samples_T{temp_str}_th{thresh_str}_{dataset_name}_{timestamp}_mahal_corrected.csv")
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Per-sample results saved: {output_file}")
    print(f"  Total samples: {len(df)}")
    print(f"  Validation: {len(df[df['split']=='validation'])}, Test: {len(df[df['split']=='test'])}")
    
    # Print stats by split
    for split in ['validation', 'test']:
        split_df = df[df['split'] == split]
        print(f"\n  {split.upper()} SET:")
        print(f"    TP: {len(split_df[split_df['result']=='TP'])}, "
              f"TN: {len(split_df[split_df['result']=='TN'])}, "
              f"FP: {len(split_df[split_df['result']=='FP'])}, "
              f"FN: {len(split_df[split_df['result']=='FN'])}")
    
    return df


def save_summary_results(optimal_temp, optimal_threshold, mahal_threshold,
                         metrics_orig_test, metrics_scaled_test,
                         calib_orig_test, calib_scaled_test,
                         dataset_name, timestamp):
    """Save summary results (TEST SET ONLY)."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    summary_file = os.path.join(OUTPUT_DIR, f"summary_temp_scaling_{dataset_name}_{timestamp}_mahal_corrected.csv")
    
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow(['=== TEMPERATURE SCALING + MAHALANOBIS (CORRECTED) ===', ''])
        writer.writerow(['Dataset', dataset_name])
        writer.writerow(['Timestamp', timestamp])
        writer.writerow(['METHODOLOGY', 'Params fitted on VALIDATION set only'])
        writer.writerow(['', ''])
        
        writer.writerow(['=== OPTIMIZED PARAMETERS (from Validation) ===', ''])
        writer.writerow(['Optimal Temperature', f"{optimal_temp:.4f}"])
        writer.writerow(['Optimal Threshold', f"{optimal_threshold:.4f}"])
        writer.writerow(['Optimal Mahalanobis Threshold', f"{mahal_threshold:.2f}"])
        writer.writerow(['Original Threshold', ORIGINAL_THRESHOLD])
        writer.writerow(['', ''])
        
        writer.writerow(['=== CLASSIFICATION (TEST SET ONLY) ===', '', ''])
        writer.writerow(['Metric', 'Original', 'Recalibrated'])
        writer.writerow(['Accuracy', f"{metrics_orig_test['accuracy']:.4f}", f"{metrics_scaled_test['accuracy']:.4f}"])
        writer.writerow(['Precision', f"{metrics_orig_test['precision']:.4f}", f"{metrics_scaled_test['precision']:.4f}"])
        writer.writerow(['Recall', f"{metrics_orig_test['recall']:.4f}", f"{metrics_scaled_test['recall']:.4f}"])
        writer.writerow(['Specificity', f"{metrics_orig_test['specificity']:.4f}", f"{metrics_scaled_test['specificity']:.4f}"])
        writer.writerow(['F1', f"{metrics_orig_test['f1']:.4f}", f"{metrics_scaled_test['f1']:.4f}"])
        writer.writerow(['', ''])
        
        writer.writerow(['=== CALIBRATION (TEST SET ONLY) ===', '', ''])
        writer.writerow(['Metric', 'Original', 'Recalibrated'])
        writer.writerow(['ECE', f"{calib_orig_test['ece']:.4f}", f"{calib_scaled_test['ece']:.4f}"])
        writer.writerow(['Confidence Gap', f"{calib_orig_test['conf_gap']:.4f}", f"{calib_scaled_test['conf_gap']:.4f}"])
    
    print(f"\n✓ Summary results saved: {summary_file}")

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_analysis(all_data, dataset_name, timestamp):
    """Run temperature scaling with proper validation/test split."""
    
    print("\n" + "=" * 70)
    print("TEMPERATURE SCALING WITH MAHALANOBIS (CORRECTED METHODOLOGY)")
    print("=" * 70)
    
    # Extract arrays
    logits = np.array([d['mean_logits'] for d in all_data])
    labels = np.array([d['label'] for d in all_data])
    
    # =========================================================================
    # Split data: 50% Validation, 50% Test
    # =========================================================================
    print("\n1. Splitting data into Validation (50%) and Test (50%) sets...")
    
    indices = np.arange(len(all_data))
    val_idx, test_idx = train_test_split(indices, test_size=0.5, random_state=42, stratify=labels)
    
    val_logits = logits[val_idx]
    val_labels = labels[val_idx]
    test_logits = logits[test_idx]
    test_labels = labels[test_idx]
    
    print(f"   Validation: {len(val_idx)} samples ({np.sum(val_labels == 0)} coughs)")
    print(f"   Test:       {len(test_idx)} samples ({np.sum(test_labels == 0)} coughs)")
    
    # =========================================================================
    # Find optimal temperature on VALIDATION set
    # =========================================================================
    print("\n2. Finding optimal temperature (VALIDATION set only)...")
    
    optimal_temp = find_optimal_temperature(val_logits, val_labels)
    print(f"   Optimal temperature: {optimal_temp:.4f}")
    
    # =========================================================================
    # Find optimal threshold on VALIDATION set
    # =========================================================================
    print("\n3. Finding optimal threshold (VALIDATION set only)...")
    
    val_probs_scaled = apply_temperature(val_logits, optimal_temp)
    optimal_threshold, best_score = find_optimal_threshold(val_probs_scaled, val_labels, metric='f1')
    print(f"   Optimal threshold: {optimal_threshold:.4f} (F1={best_score:.4f})")
    
    # =========================================================================
    # Compute Mahalanobis params on VALIDATION TPs only (CORRECTED)
    # =========================================================================
    print("\n4. Computing Mahalanobis params (VALIDATION TPs only)...")
    
    tp_mean, tp_precision = compute_mahalanobis_params_from_validation(
        all_data, val_idx, optimal_temp, optimal_threshold)
    
    if tp_mean is None:
        print("   ⚠️ Skipping Mahalanobis (not enough validation TPs)")
        mahal_threshold = 40.0
    else:
        # Compute distances for ALL samples using validation-fitted params
        all_data = compute_mahalanobis_distances_for_all(all_data, tp_mean, tp_precision)
        
        # Find optimal Mahalanobis threshold on VALIDATION set
        mahal_threshold = find_optimal_mahalanobis_threshold(
            all_data, val_idx, optimal_temp, optimal_threshold)
    
    # =========================================================================
    # Evaluate on TEST set (never seen during fitting)
    # =========================================================================
    print("\n5. Evaluating on TEST set (held out)...")
    
    # Original (T=1, threshold=0.7)
    test_probs_original = apply_temperature(test_logits, 1.0)
    metrics_orig_test = compute_metrics(test_probs_original, test_labels, ORIGINAL_THRESHOLD)
    calib_orig_test = compute_calibration_metrics(test_probs_original, test_labels, metrics_orig_test['predictions'])
    
    # Recalibrated (T=optimal, threshold=optimal)
    test_probs_scaled = apply_temperature(test_logits, optimal_temp)
    metrics_scaled_test = compute_metrics(test_probs_scaled, test_labels, optimal_threshold)
    calib_scaled_test = compute_calibration_metrics(test_probs_scaled, test_labels, metrics_scaled_test['predictions'])
    
    # =========================================================================
    # Print Results (TEST SET ONLY)
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS (TEST SET ONLY - NOT USED FOR FITTING)")
    print("=" * 70)
    
    print(f"\n--- Optimized Parameters (from Validation) ---")
    print(f"Temperature: {optimal_temp:.4f}")
    print(f"Threshold: {optimal_threshold:.4f}")
    print(f"Mahalanobis Threshold: {mahal_threshold:.2f}")
    
    print(f"\n--- Classification Metrics (TEST SET) ---")
    print(f"{'Metric':<15} {'Original':>15} {'Recalibrated':>15}")
    print("-" * 50)
    
    for metric in ['accuracy', 'precision', 'recall', 'specificity', 'f1']:
        o = metrics_orig_test[metric]
        r = metrics_scaled_test[metric]
        print(f"{metric.capitalize():<15} {o:>14.1%} {r:>15.1%}")
    
    print(f"\n--- Calibration (TEST SET) ---")
    print(f"{'Metric':<25} {'Original':>12} {'Recalibrated':>12}")
    print("-" * 52)
    print(f"{'ECE':<25} {calib_orig_test['ece']:>12.4f} {calib_scaled_test['ece']:>12.4f}")
    print(f"{'Confidence Gap':<25} {calib_orig_test['conf_gap']:>12.4f} {calib_scaled_test['conf_gap']:>12.4f}")
    
    # =========================================================================
    # Mahalanobis Analysis (TEST SET ONLY)
    # =========================================================================
    if tp_mean is not None:
        print(f"\n--- Mahalanobis Analysis (TEST SET) ---")
        
        test_data = [all_data[i] for i in test_idx]
        test_distances = np.array([d['mahalanobis_distance'] for d in test_data])
        
        test_results = []
        for i, idx in enumerate(test_idx):
            pred = metrics_scaled_test['predictions'][i]
            true = test_labels[i]
            if pred == 0 and true == 0:
                test_results.append('TP')
            elif pred == 1 and true == 1:
                test_results.append('TN')
            elif pred == 0 and true == 1:
                test_results.append('FP')
            else:
                test_results.append('FN')
        
        test_results = np.array(test_results)
        
        print(f"\n   Distance by Result Type:")
        for result in ['TP', 'FP', 'TN', 'FN']:
            mask = test_results == result
            if mask.sum() > 0:
                mean_d = test_distances[mask].mean()
                std_d = test_distances[mask].std()
                print(f"     {result}: mean={mean_d:.2f}, std={std_d:.2f}, n={mask.sum()}")
        
        print(f"\n   Rejection Analysis (TEST SET):")
        for thresh in [35, 40, 45]:
            tp_mask = test_results == 'TP'
            fp_mask = test_results == 'FP'
            
            tp_kept = (test_distances[tp_mask] <= thresh).sum()
            tp_total = tp_mask.sum()
            fp_rejected = (test_distances[fp_mask] > thresh).sum()
            fp_total = fp_mask.sum()
            
            if tp_total > 0 and fp_total > 0:
                print(f"     Threshold {thresh}: TPs kept={100*tp_kept/tp_total:.1f}%, FPs rejected={100*fp_rejected/fp_total:.1f}%")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    save_summary_results(
        optimal_temp, optimal_threshold, mahal_threshold,
        metrics_orig_test, metrics_scaled_test,
        calib_orig_test, calib_scaled_test,
        dataset_name, timestamp
    )
    
    save_per_sample_results(
        all_data, val_idx, test_idx,
        optimal_temp, optimal_threshold, mahal_threshold,
        dataset_name, timestamp
    )
    
    return optimal_temp, optimal_threshold, mahal_threshold

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Temperature scaling with Mahalanobis (CORRECTED)')
    parser.add_argument('dataset', choices=['esc50', 'coswara'],
                        help='Which dataset to process')
    
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("CoughGuard - Temperature Scaling + Mahalanobis (CORRECTED)")
    print("=" * 70)
    print(f"\nDataset: {args.dataset.upper()}")
    print(f"\nMETHODOLOGY (Corrected):")
    print(f"  1. Split: 50% Validation, 50% Test (stratified)")
    print(f"  2. VALIDATION ONLY: Optimize T, threshold, Mahalanobis params")
    print(f"  3. TEST ONLY: Final evaluation (never touched during fitting)")
    
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model not found: {MODEL_PATH}")
        return
    
    print("\n" + "-" * 70)
    print("Loading model...")
    model = load_model()
    print("✓ Model loaded!")
    
    print("\nCollecting data (with MC Dropout stats + embeddings)...")
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
