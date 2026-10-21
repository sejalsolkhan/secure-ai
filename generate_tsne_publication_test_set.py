"""
Publication-Quality t-SNE for ISEF Presentation (CORRECTED)
============================================================

Uses TEST SET embeddings only with corrected Mahalanobis thresholds.

Usage:
    python generate_tsne_publication_corrected.py --input embeddings_test_set/

Requires:
    - embeddings.npy (from extract_embeddings_test_set.py)
    - embedding_metadata.csv (with result, category, mahalanobis_distance)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Corrected threshold (from test set analysis)
MAHALANOBIS_THRESHOLD = 60  # 94.8% TPs kept, 53.9% FPs rejected

OUTPUT_DIR = "."
PERPLEXITY = 30
RANDOM_SEED = 42

# =============================================================================
# STYLE SETTINGS
# =============================================================================

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color scheme
COLORS = {
    'TP': '#27ae60',  # Green
    'FP': '#e74c3c',  # Red  
    'TN': '#95a5a6',  # Gray
    'FN': '#f39c12',  # Orange
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_data(input_dir):
    """Load embeddings and metadata."""
    
    emb_path = os.path.join(input_dir, 'embeddings.npy')
    meta_path = os.path.join(input_dir, 'embedding_metadata.csv')
    
    print(f"  Loading: {emb_path}")
    embeddings = np.load(emb_path)
    
    print(f"  Loading: {meta_path}")
    metadata = pd.read_csv(meta_path)
    
    print(f"  Shape: {embeddings.shape}")
    print(f"  Distribution: {dict(metadata['result'].value_counts())}")
    
    return embeddings, metadata


def compute_tsne(embeddings, perplexity=30, seed=42):
    """Compute t-SNE coordinates."""
    print(f"  Computing t-SNE (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, n_iter=1000)
    coords = tsne.fit_transform(embeddings)
    return coords

# =============================================================================
# PLOT 1: Main Visualization with Inset
# =============================================================================

def plot_main_with_inset(coords, metadata, output_path):
    """
    Main t-SNE plot with:
    - All result types colored
    - Zoomed inset showing TP cluster
    - Annotations
    """
    
    labels = metadata['result'].values
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot order: TN first (background), then others on top
    plot_order = ['TN', 'FN', 'FP', 'TP']
    sizes = {'TN': 8, 'FN': 25, 'FP': 40, 'TP': 40}
    alphas = {'TN': 0.15, 'FN': 0.6, 'FP': 0.8, 'TP': 0.8}
    zorders = {'TN': 1, 'FN': 2, 'FP': 3, 'TP': 4}
    
    for result in plot_order:
        mask = labels == result
        if mask.sum() > 0:
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=COLORS[result], label=f'{result} (n={mask.sum()})',
                       s=sizes[result], alpha=alphas[result], 
                       edgecolors='white' if result in ['TP', 'FP'] else 'none',
                       linewidths=0.3, zorder=zorders[result])
    
    # --- ZOOMED INSET for TP cluster ---
    tp_mask = labels == 'TP'
    if tp_mask.sum() > 10:
        tp_coords = coords[tp_mask]
        tp_center = tp_coords.mean(axis=0)
        tp_spread = tp_coords.std(axis=0).max() * 2.5
        
        # Create inset
        axins = inset_axes(ax, width="30%", height="30%", loc='lower left',
                           bbox_to_anchor=(0.02, 0.02, 1, 1), bbox_transform=ax.transAxes)
        
        # Plot TPs in inset
        axins.scatter(tp_coords[:, 0], tp_coords[:, 1],
                      c=COLORS['TP'], s=50, alpha=0.9, edgecolors='white', linewidths=0.5)
        
        # Set inset bounds
        axins.set_xlim(tp_center[0] - tp_spread, tp_center[0] + tp_spread)
        axins.set_ylim(tp_center[1] - tp_spread, tp_center[1] + tp_spread)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_title('TP Cluster', fontsize=9, fontweight='bold')
        
        # Draw connector
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle='--')
    
    # --- Annotations ---
    tp_center = coords[tp_mask].mean(axis=0)
    ax.annotate('True Positives\n(low Mahalanobis)',
                xy=tp_center,
                xytext=(40, 40), textcoords='offset points',
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    
    fp_mask = labels == 'FP'
    if fp_mask.sum() > 0:
        fp_center = coords[fp_mask].mean(axis=0)
        ax.annotate('False Positives\n(high Mahalanobis)',
                    xy=fp_center,
                    xytext=(-50, -50), textcoords='offset points',
                    fontsize=9, ha='right',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('Embedding Space: Cough Detection (Test Set)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")

# =============================================================================
# PLOT 2: t-SNE + Mahalanobis Histogram Panel
# =============================================================================

def plot_mahalanobis_panel(coords, metadata, threshold, output_path):
    """
    Two-panel figure:
    - Left: t-SNE with TP/FP only
    - Right: Mahalanobis distance histogram with threshold
    """
    
    labels = metadata['result'].values
    mahal_distances = metadata['mahalanobis_distance'].values
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Left: t-SNE ---
    ax = axes[0]
    tp_mask = labels == 'TP'
    fp_mask = labels == 'FP'
    other_mask = ~(tp_mask | fp_mask)
    
    ax.scatter(coords[other_mask, 0], coords[other_mask, 1],
               c='#ecf0f1', s=8, alpha=0.15, label='TN/FN')
    ax.scatter(coords[fp_mask, 0], coords[fp_mask, 1],
               c=COLORS['FP'], s=45, alpha=0.8, 
               edgecolors='white', linewidths=0.5,
               label=f'False Positives (n={fp_mask.sum()})')
    ax.scatter(coords[tp_mask, 0], coords[tp_mask, 1],
               c=COLORS['TP'], s=45, alpha=0.8,
               edgecolors='white', linewidths=0.5,
               label=f'True Positives (n={tp_mask.sum()})')
    
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('Embedding Space', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # --- Right: Mahalanobis histogram ---
    ax = axes[1]
    
    tp_dist = mahal_distances[tp_mask]
    fp_dist = mahal_distances[fp_mask]
    
    # Histograms
    max_dist = max(np.nanmax(tp_dist), np.nanmax(fp_dist))
    bins = np.linspace(0, min(max_dist, 150), 40)
    
    ax.hist(tp_dist, bins=bins, alpha=0.7, color=COLORS['TP'], 
            label=f'TP (μ={np.nanmean(tp_dist):.1f})', density=True)
    ax.hist(fp_dist, bins=bins, alpha=0.7, color=COLORS['FP'], 
            label=f'FP (μ={np.nanmean(fp_dist):.1f})', density=True)
    
    # Threshold line
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2, 
               label=f'Threshold = {threshold}')
    
    # Shade rejection region
    ax.axvspan(threshold, bins[-1], alpha=0.1, color='red')
    
    # Stats annotation
    fp_rejected = np.sum(fp_dist > threshold) / len(fp_dist) * 100
    tp_kept = np.sum(tp_dist <= threshold) / len(tp_dist) * 100
    
    ax.annotate(f'TPs Kept: {tp_kept:.1f}%\nFPs Rejected: {fp_rejected:.1f}%',
                xy=(threshold + 3, ax.get_ylim()[1] * 0.75),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray'))
    
    ax.set_xlabel('Mahalanobis Distance')
    ax.set_ylabel('Density')
    ax.set_title('Mahalanobis Distance Distribution', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    
    plt.suptitle('Multi-Signal Uncertainty Quantification (Test Set)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")

# =============================================================================
# PLOT 3: Multi-Threshold Comparison
# =============================================================================

def plot_threshold_comparison(coords, metadata, output_path):
    """
    Show effect of different thresholds on TP/FP classification.
    """
    
    labels = metadata['result'].values
    mahal_distances = metadata['mahalanobis_distance'].values
    
    thresholds = [55, 60, 65]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    tp_mask = labels == 'TP'
    fp_mask = labels == 'FP'
    
    for ax, thresh in zip(axes, thresholds):
        # Classify based on threshold
        tp_kept_mask = tp_mask & (mahal_distances <= thresh)
        tp_rejected_mask = tp_mask & (mahal_distances > thresh)
        fp_kept_mask = fp_mask & (mahal_distances <= thresh)
        fp_rejected_mask = fp_mask & (mahal_distances > thresh)
        other_mask = ~(tp_mask | fp_mask)
        
        # Plot
        ax.scatter(coords[other_mask, 0], coords[other_mask, 1],
                   c='#ecf0f1', s=8, alpha=0.1)
        
        # FPs
        ax.scatter(coords[fp_rejected_mask, 0], coords[fp_rejected_mask, 1],
                   c=COLORS['FP'], s=35, alpha=0.8, marker='x',
                   label=f'FP rejected ({fp_rejected_mask.sum()})')
        ax.scatter(coords[fp_kept_mask, 0], coords[fp_kept_mask, 1],
                   c=COLORS['FP'], s=35, alpha=0.4,
                   label=f'FP kept ({fp_kept_mask.sum()})')
        
        # TPs
        ax.scatter(coords[tp_rejected_mask, 0], coords[tp_rejected_mask, 1],
                   c=COLORS['TP'], s=35, alpha=0.8, marker='x',
                   label=f'TP lost ({tp_rejected_mask.sum()})')
        ax.scatter(coords[tp_kept_mask, 0], coords[tp_kept_mask, 1],
                   c=COLORS['TP'], s=35, alpha=0.8,
                   label=f'TP kept ({tp_kept_mask.sum()})')
        
        # Stats
        tp_rate = tp_kept_mask.sum() / tp_mask.sum() * 100
        fp_rate = fp_rejected_mask.sum() / fp_mask.sum() * 100
        
        ax.set_title(f'Threshold = {thresh}\nTPs: {tp_rate:.0f}% kept, FPs: {fp_rate:.0f}% rejected',
                     fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Mahalanobis Threshold Operating Points (Test Set)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")

# =============================================================================
# PLOT 4: FP Categories
# =============================================================================

def plot_fp_categories(coords, metadata, output_path):
    """FP breakdown by sound category."""
    
    labels = metadata['result'].values
    categories = metadata['category'].values
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    fp_mask = labels == 'FP'
    non_fp_mask = ~fp_mask
    
    # Background
    ax.scatter(coords[non_fp_mask, 0], coords[non_fp_mask, 1],
               c='#ecf0f1', s=8, alpha=0.1)
    
    # FP categories
    fp_categories = categories[fp_mask]
    unique_cats = sorted(set(fp_categories))
    cmap = plt.cm.tab10
    
    for i, cat in enumerate(unique_cats[:10]):  # Limit to 10 categories
        mask = (labels == 'FP') & (categories == cat)
        if mask.sum() > 0:
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[cmap(i % 10)], s=50, alpha=0.8,
                       edgecolors='white', linewidths=0.5,
                       label=f'{cat} (n={mask.sum()})')
    
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('False Positive Categories in Embedding Space', fontsize=13, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Publication-quality t-SNE (TEST SET ONLY)')
    parser.add_argument('--input', type=str, default='embeddings_test_set',
                        help='Directory with embeddings.npy and embedding_metadata.csv')
    parser.add_argument('--output', type=str, default='.',
                        help='Output directory for plots')
    parser.add_argument('--threshold', type=float, default=MAHALANOBIS_THRESHOLD,
                        help='Mahalanobis threshold for visualization')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PUBLICATION-QUALITY t-SNE (TEST SET ONLY)")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading embeddings...")
    embeddings, metadata = load_data(args.input)
    
    # Compute t-SNE
    print("\n[2] Computing t-SNE...")
    coords = compute_tsne(embeddings, PERPLEXITY, RANDOM_SEED)
    
    # Save coordinates for reuse
    np.save(os.path.join(args.input, 'tsne_coords.npy'), coords)
    print(f"  ✓ Saved: {args.input}/tsne_coords.npy")
    
    # Generate plots
    print("\n[3] Generating plots...")
    
    os.makedirs(args.output, exist_ok=True)
    
    # Plot 1: Main with inset
    plot_main_with_inset(
        coords, metadata,
        os.path.join(args.output, 'tsne_main_annotated.png')
    )
    
    # Plot 2: Mahalanobis panel
    plot_mahalanobis_panel(
        coords, metadata, args.threshold,
        os.path.join(args.output, 'tsne_mahalanobis_panel.png')
    )
    
    # Plot 3: Threshold comparison
    plot_threshold_comparison(
        coords, metadata,
        os.path.join(args.output, 'tsne_threshold_comparison.png')
    )
    
    # Plot 4: FP categories
    plot_fp_categories(
        coords, metadata,
        os.path.join(args.output, 'tsne_fp_categories.png')
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    print(f"""
  1. tsne_main_annotated.png       - Main visualization with TP cluster inset
  2. tsne_mahalanobis_panel.png    - t-SNE + Mahalanobis histogram side-by-side
  3. tsne_threshold_comparison.png - Effect of different thresholds (55, 60, 65)
  4. tsne_fp_categories.png        - FP breakdown by sound category
""")
    print("✓ Done!")


if __name__ == '__main__':
    main()
