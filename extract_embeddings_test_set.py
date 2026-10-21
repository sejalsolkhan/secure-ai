"""
Embedding Extraction for t-SNE Visualization (TEST SET ONLY)
=============================================================

Extracts embeddings from TEST SET samples only for publication-quality
t-SNE visualization.

Usage:
    python extract_embeddings_test_set.py \
        --csv outputs/all_samples_T8p57_th0p54_coswara_20260318_145415_mahal_corrected.csv \
        --model models/mn/mn10_mydata_epoch_13.pt \
        --output embeddings_test_set/
"""

import sys
sys.path.insert(0, '.')

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION (same as your inference code)
# =============================================================================

SAMPLE_RATE = 32000
N_FFT = 1024
HOP_LENGTH = 320
N_MELS = 128
WIN_LENGTH = 800
FMIN = 0.0
FMAX = 15000

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
# MODEL LOADING
# =============================================================================

def load_model(model_path: str):
    """Load the model."""
    from models.mn.model import get_model
    
    model = get_model(
        num_classes=2,
        pretrained_name=None,
        width_mult=1.0,
        head_type="mlp",
        se_dims='c'
    )
    
    state_dict = torch.load(model_path, map_location='cpu')
    
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model

# =============================================================================
# EMBEDDING EXTRACTION
# =============================================================================

def extract_embeddings(model, df: pd.DataFrame, max_per_result: int = None):
    """
    Extract embeddings for samples in dataframe.
    
    Args:
        model: Loaded model
        df: DataFrame with 'file_path', 'result', 'category', 'mahalanobis_distance'
        max_per_result: Max samples per result type (for faster extraction)
    
    Returns:
        embeddings, metadata_df
    """
    
    embeddings = []
    metadata = []
    
    # Process each result type
    for result in ['TP', 'FP', 'TN', 'FN']:
        subset = df[df['result'] == result].copy()
        
        if max_per_result and len(subset) > max_per_result:
            subset = subset.sample(n=max_per_result, random_state=42)
        
        print(f"  Processing {len(subset)} {result} samples...")
        
        success_count = 0
        for idx, row in subset.iterrows():
            file_path = row['file_path']
            
            if not os.path.exists(file_path):
                continue
            
            try:
                input_tensor = preprocess_audio(file_path)
                
                with torch.no_grad():
                    logits, features = model(input_tensor)
                
                embedding = features.cpu().numpy().flatten()
                
                embeddings.append(embedding)
                metadata.append({
                    'file_path': file_path,
                    'result': result,
                    'category': row['category'],
                    'mahalanobis_distance': row['mahalanobis_distance'],
                    'cough_prob_calibrated': row.get('cough_prob_calibrated', np.nan),
                    'epistemic_uncertainty': row.get('epistemic_uncertainty', np.nan),
                })
                
                success_count += 1
                
            except Exception as e:
                pass
        
        print(f"    ✓ Extracted {success_count} embeddings")
    
    return np.array(embeddings), pd.DataFrame(metadata)

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Extract embeddings for t-SNE (TEST SET ONLY)')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to corrected all_samples CSV (with split column)')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model .pt file')
    parser.add_argument('--output', type=str, default='embeddings_test_set',
                        help='Output directory')
    parser.add_argument('--max_per_result', type=int, default=None,
                        help='Max samples per result type (None = all)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EMBEDDING EXTRACTION (TEST SET ONLY)")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load CSV
    print(f"\nLoading: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"Total samples: {len(df)}")
    
    # Filter to TEST SET ONLY
    if 'split' in df.columns:
        df_test = df[df['split'] == 'test'].copy()
        print(f"Test set samples: {len(df_test)}")
    else:
        print("WARNING: No 'split' column found - using all data")
        df_test = df
    
    print(f"\nResult distribution (test set):")
    print(df_test['result'].value_counts().to_string())
    
    # Load model
    print(f"\nLoading model: {args.model}")
    model = load_model(args.model)
    print("✓ Model loaded")
    
    # Extract embeddings
    print(f"\nExtracting embeddings...")
    start_time = datetime.now()
    
    embeddings, metadata_df = extract_embeddings(
        model, df_test, max_per_result=args.max_per_result
    )
    
    elapsed = datetime.now() - start_time
    print(f"\n✓ Extracted {len(embeddings)} embeddings in {elapsed}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    
    # Save
    emb_path = os.path.join(args.output, 'embeddings.npy')
    meta_path = os.path.join(args.output, 'embedding_metadata.csv')
    
    np.save(emb_path, embeddings)
    metadata_df.to_csv(meta_path, index=False)
    
    print(f"\n✓ Saved: {emb_path}")
    print(f"✓ Saved: {meta_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nResult distribution in extracted embeddings:")
    print(metadata_df['result'].value_counts().to_string())
    
    print(f"\nNext step: Run t-SNE visualization")
    print(f"  python generate_tsne_publication_corrected.py --input {args.output}")


if __name__ == '__main__':
    main()
