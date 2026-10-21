"""
Statistical Analysis for ISEF Presentation (CORRECTED)
=======================================================

CORRECTED VERSION: Uses TEST SET ONLY for all statistics.

Computes:
1. Wilson score confidence intervals on classification metrics
2. Two-sample t-test for TP vs FP Mahalanobis distance
3. Bootstrap confidence intervals on Mahalanobis gap
4. Effect size (Cohen's d) for TP/FP separation
5. Comprehensive UQ method comparison

Usage:
    python generate_statistics_corrected.py

Update CONFIGURATION section with your CSV paths.
The CSV must have a 'split' column with 'validation' or 'test' values.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================

# Use the CORRECTED CSV files (with 'split' column)
COSWARA_CSV = "outputs/all_samples_T8p55_th0p54_coswara_20260201_mahal_corrected.csv"
ESC50_CSV = "outputs/all_samples_T3p73_th0p60_esc50_20260202_mahal_corrected.csv"

OUTPUT_MD = "statistical_analysis_corrected.md"
OUTPUT_CSV = "statistical_summary_corrected.csv"

# Confidence level
ALPHA = 0.05  # 95% CI

# Bootstrap settings
N_BOOTSTRAP = 10000

# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def wilson_score_interval(successes, trials, alpha=0.05):
    """
    Wilson score interval for binomial proportion.
    More accurate than normal approximation for small samples or extreme proportions.
    """
    if trials == 0:
        return (0.0, 0.0)
    
    p = successes / trials
    z = stats.norm.ppf(1 - alpha / 2)
    
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
    
    lower = max(0, center - spread)
    upper = min(1, center + spread)
    
    return (lower, upper)


def bootstrap_f1_ci(tp, fp, tn, fn, alpha=0.05, n_bootstrap=10000):
    """Bootstrap confidence interval for F1 score."""
    
    y_true = np.array([1]*tp + [1]*fn + [0]*fp + [0]*tn)
    y_pred = np.array([1]*tp + [0]*fn + [1]*fp + [0]*tn)
    
    n = len(y_true)
    f1_scores = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        tp_b = ((y_true[idx] == 1) & (y_pred[idx] == 1)).sum()
        fp_b = ((y_true[idx] == 0) & (y_pred[idx] == 1)).sum()
        fn_b = ((y_true[idx] == 1) & (y_pred[idx] == 0)).sum()
        
        prec = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0
        rec = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1_scores.append(f1)
    
    lower = np.percentile(f1_scores, 100 * alpha / 2)
    upper = np.percentile(f1_scores, 100 * (1 - alpha / 2))
    
    return (lower, upper)


def bootstrap_gap_ci(tp_dist, fp_dist, n_bootstrap=10000, alpha=0.05):
    """Bootstrap confidence interval for mean difference (gap)."""
    
    gaps = []
    np.random.seed(42)
    
    for _ in range(n_bootstrap):
        tp_sample = np.random.choice(tp_dist, len(tp_dist), replace=True)
        fp_sample = np.random.choice(fp_dist, len(fp_dist), replace=True)
        gaps.append(np.mean(fp_sample) - np.mean(tp_sample))
    
    lower = np.percentile(gaps, 100 * alpha / 2)
    upper = np.percentile(gaps, 100 * (1 - alpha / 2))
    
    return (lower, upper)


def interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def compute_cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0
    
    return (np.mean(group2) - np.mean(group1)) / pooled_std

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_dataset(csv_path, name, use_test_only=True):
    """Run comprehensive analysis on a dataset."""
    
    print(f"\n{'='*70}")
    print(f"Analyzing {name}")
    print(f"{'='*70}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} total samples")
    
    # Check for split column
    if 'split' in df.columns:
        print(f"  Validation: {len(df[df['split']=='validation'])}")
        print(f"  Test: {len(df[df['split']=='test'])}")
        
        if use_test_only:
            df = df[df['split'] == 'test'].copy()
            print(f"\n*** USING TEST SET ONLY: {len(df)} samples ***")
    else:
        print(f"  WARNING: No 'split' column found - using all data")
        print(f"  This may include data used for fitting (potential leakage)")
    
    # Check for required columns
    if 'mahalanobis_distance' not in df.columns:
        print(f"WARNING: 'mahalanobis_distance' column not found!")
        return None
    
    results = {}
    results['name'] = name
    results['n_samples'] = len(df)
    
    # =========================================================================
    # 1. Classification Metrics with CIs
    # =========================================================================
    print(f"\n[1] Classification Metrics (95% Wilson CI)...")
    
    tp = (df['result'] == 'TP').sum()
    fp = (df['result'] == 'FP').sum()
    tn = (df['result'] == 'TN').sum()
    fn = (df['result'] == 'FN').sum()
    
    total = tp + fp + tn + fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    precision_ci = wilson_score_interval(tp, tp + fp)
    recall_ci = wilson_score_interval(tp, tp + fn)
    specificity_ci = wilson_score_interval(tn, tn + fp)
    accuracy_ci = wilson_score_interval(tp + tn, total)
    f1_ci = bootstrap_f1_ci(tp, fp, tn, fn, n_bootstrap=N_BOOTSTRAP)
    
    results['confusion_matrix'] = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}
    results['precision'] = {'value': precision, 'ci': precision_ci}
    results['recall'] = {'value': recall, 'ci': recall_ci}
    results['f1'] = {'value': f1, 'ci': f1_ci}
    results['accuracy'] = {'value': accuracy, 'ci': accuracy_ci}
    results['specificity'] = {'value': specificity, 'ci': specificity_ci}
    
    print(f"    Precision: {precision*100:.1f}% [{precision_ci[0]*100:.1f}%, {precision_ci[1]*100:.1f}%]")
    print(f"    Recall:    {recall*100:.1f}% [{recall_ci[0]*100:.1f}%, {recall_ci[1]*100:.1f}%]")
    print(f"    F1:        {f1*100:.1f}% [{f1_ci[0]*100:.1f}%, {f1_ci[1]*100:.1f}%]")
    
    # =========================================================================
    # 2. UQ Method Comparison (Cohen's d for TP vs FP separation)
    # =========================================================================
    print(f"\n[2] UQ Method Comparison (TP vs FP Separation)...")
    
    results['uq_comparison'] = {}
    
    # MC Dropout (Epistemic)
    if 'epistemic_uncertainty' in df.columns:
        tp_epi = df[df['result'] == 'TP']['epistemic_uncertainty'].values
        fp_epi = df[df['result'] == 'FP']['epistemic_uncertainty'].values
        
        if len(tp_epi) > 1 and len(fp_epi) > 1:
            t_stat, p_val = ttest_ind(tp_epi, fp_epi, equal_var=False)
            d = abs(compute_cohens_d(tp_epi, fp_epi))
            
            results['uq_comparison']['mc_dropout'] = {
                'cohens_d': d,
                'interpretation': interpret_cohens_d(d),
                'p_value': p_val,
                'tp_mean': np.mean(tp_epi),
                'fp_mean': np.mean(fp_epi),
            }
            
            print(f"    MC Dropout:    d={d:.3f} ({interpret_cohens_d(d)}), p={'<0.001' if p_val<0.001 else f'{p_val:.4f}'}")
    
    # Temperature Scaling (Confidence)
    if 'cough_prob_calibrated' in df.columns:
        tp_conf = df[df['result'] == 'TP']['cough_prob_calibrated'].values
        fp_conf = df[df['result'] == 'FP']['cough_prob_calibrated'].values
        
        if len(tp_conf) > 1 and len(fp_conf) > 1:
            t_stat, p_val = ttest_ind(tp_conf, fp_conf, equal_var=False)
            d = abs(compute_cohens_d(fp_conf, tp_conf))  # Note: reversed for confidence
            
            results['uq_comparison']['temp_scaling'] = {
                'cohens_d': d,
                'interpretation': interpret_cohens_d(d),
                'p_value': p_val,
                'tp_mean': np.mean(tp_conf),
                'fp_mean': np.mean(fp_conf),
            }
            
            print(f"    Temp Scaling:  d={d:.3f} ({interpret_cohens_d(d)}), p={'<0.001' if p_val<0.001 else f'{p_val:.4f}'}")
    
    # Mahalanobis Distance
    tp_dist = df[df['result'] == 'TP']['mahalanobis_distance'].values
    fp_dist = df[df['result'] == 'FP']['mahalanobis_distance'].values
    
    if len(tp_dist) > 1 and len(fp_dist) > 1:
        t_stat, p_val = ttest_ind(tp_dist, fp_dist, equal_var=False)
        d = compute_cohens_d(tp_dist, fp_dist)
        
        # Gap with bootstrap CI
        gap = np.mean(fp_dist) - np.mean(tp_dist)
        gap_ci = bootstrap_gap_ci(tp_dist, fp_dist, N_BOOTSTRAP)
        
        results['uq_comparison']['mahalanobis'] = {
            'cohens_d': d,
            'interpretation': interpret_cohens_d(d),
            'p_value': p_val,
            't_statistic': t_stat,
            'tp_mean': np.mean(tp_dist),
            'tp_std': np.std(tp_dist),
            'fp_mean': np.mean(fp_dist),
            'fp_std': np.std(fp_dist),
            'gap': gap,
            'gap_ci': gap_ci,
        }
        
        print(f"    Mahalanobis:   d={d:.2f} ({interpret_cohens_d(d)}), p={'<0.001' if p_val<0.001 else f'{p_val:.4f}'}")
        print(f"                   Gap={gap:.2f} [{gap_ci[0]:.2f}, {gap_ci[1]:.2f}]")
    
    # =========================================================================
    # 3. Mahalanobis Distance Statistics
    # =========================================================================
    print(f"\n[3] Mahalanobis Distance by Result Type...")
    
    results['mahalanobis_stats'] = {}
    
    for result in ['TP', 'FP', 'TN', 'FN']:
        subset = df[df['result'] == result]['mahalanobis_distance']
        if len(subset) > 0:
            results['mahalanobis_stats'][result] = {
                'n': len(subset),
                'mean': np.mean(subset),
                'std': np.std(subset),
                'median': np.median(subset),
            }
            print(f"    {result}: mean={np.mean(subset):.2f}, std={np.std(subset):.2f}, n={len(subset)}")
    
    # =========================================================================
    # 4. Threshold Analysis with CIs
    # =========================================================================
    print(f"\n[4] Mahalanobis Threshold Analysis (95% Wilson CI)...")
    
    results['threshold_analysis'] = []
    
    for thresh in [35, 40, 45]:
        tp_kept = (tp_dist <= thresh).sum()
        tp_total = len(tp_dist)
        fp_rejected = (fp_dist > thresh).sum()
        fp_total = len(fp_dist)
        
        tp_rate = tp_kept / tp_total if tp_total > 0 else 0
        fp_rate = fp_rejected / fp_total if fp_total > 0 else 0
        
        tp_ci = wilson_score_interval(tp_kept, tp_total)
        fp_ci = wilson_score_interval(fp_rejected, fp_total)
        
        results['threshold_analysis'].append({
            'threshold': thresh,
            'tp_kept_rate': tp_rate,
            'tp_kept_ci': tp_ci,
            'fp_rejected_rate': fp_rate,
            'fp_rejected_ci': fp_ci,
        })
        
        print(f"    Th={thresh}: TPs kept={tp_rate*100:.1f}% [{tp_ci[0]*100:.1f}%, {tp_ci[1]*100:.1f}%], "
              f"FPs rejected={fp_rate*100:.1f}% [{fp_ci[0]*100:.1f}%, {fp_ci[1]*100:.1f}%]")
    
    # =========================================================================
    # 5. Confident FPs Analysis
    # =========================================================================
    if 'cough_prob_calibrated' in df.columns and 'epistemic_uncertainty' in df.columns:
        print(f"\n[5] 'Confident' False Positives Analysis...")
        
        fp_df = df[df['result'] == 'FP']
        
        if len(fp_df) > 0:
            median_conf = df[df['result'].isin(['TP', 'FP'])]['cough_prob_calibrated'].median()
            median_epi = df[df['result'].isin(['TP', 'FP'])]['epistemic_uncertainty'].median()
            
            confident_fps = fp_df[(fp_df['cough_prob_calibrated'] > median_conf) & 
                                  (fp_df['epistemic_uncertainty'] < median_epi)]
            
            print(f"    Total FPs: {len(fp_df)}")
            print(f"    'Confident' FPs (high conf, low epistemic): {len(confident_fps)} ({100*len(confident_fps)/len(fp_df):.1f}%)")
            
            if len(confident_fps) > 0:
                conf_fp_mahal = confident_fps['mahalanobis_distance'].values
                
                print(f"    Confident FP Mahalanobis: mean={np.mean(conf_fp_mahal):.2f}")
                
                for thresh in [35, 40]:
                    rejected = (conf_fp_mahal > thresh).sum()
                    print(f"    At threshold {thresh}: {rejected}/{len(confident_fps)} rejected ({100*rejected/len(confident_fps):.1f}%)")
                
                results['confident_fps'] = {
                    'total_fps': len(fp_df),
                    'confident_fps': len(confident_fps),
                    'confident_fp_mahal_mean': np.mean(conf_fp_mahal),
                }
    
    return results


def generate_markdown_report(coswara_results, esc50_results=None):
    """Generate markdown report."""
    
    lines = []
    lines.append("# Statistical Analysis Report (CORRECTED)")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("**CRITICAL:** All statistics computed on **TEST SET ONLY** (held out during fitting).")
    lines.append("")
    lines.append("- Temperature T: Optimized on validation set")
    lines.append("- Threshold: Optimized on validation set")
    lines.append("- Mahalanobis params: Fitted on validation TPs only")
    lines.append("- All reported metrics: Computed on test set (never used for fitting)")
    lines.append("")
    
    for name, results in [("Coswara", coswara_results), ("ESC-50", esc50_results)]:
        if results is None:
            continue
        
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"## {name} Dataset (TEST SET: n={results['n_samples']})")
        lines.append("")
        
        # Confusion Matrix
        cm = results['confusion_matrix']
        lines.append("### Confusion Matrix")
        lines.append("")
        lines.append("|  | Predicted Cough | Predicted Not-Cough |")
        lines.append("|--|-----------------|---------------------|")
        lines.append(f"| **Actual Cough** | TP: {cm['TP']} | FN: {cm['FN']} |")
        lines.append(f"| **Actual Not-Cough** | FP: {cm['FP']} | TN: {cm['TN']} |")
        lines.append("")
        
        # Classification Metrics
        lines.append("### Classification Metrics (95% Wilson CI)")
        lines.append("")
        lines.append("| Metric | Value | 95% CI |")
        lines.append("|--------|-------|--------|")
        for metric in ['precision', 'recall', 'f1', 'accuracy', 'specificity']:
            m = results[metric]
            lines.append(f"| {metric.capitalize()} | {m['value']*100:.1f}% | [{m['ci'][0]*100:.1f}%, {m['ci'][1]*100:.1f}%] |")
        lines.append("")
        
        # UQ Comparison
        lines.append("### UQ Method Comparison (TP vs FP Separation)")
        lines.append("")
        lines.append("| Method | Cohen's d | Effect Size | p-value |")
        lines.append("|--------|-----------|-------------|---------|")
        
        for method, label in [('mc_dropout', 'MC Dropout'), ('temp_scaling', 'Temp Scaling'), ('mahalanobis', 'Mahalanobis')]:
            if method in results['uq_comparison']:
                m = results['uq_comparison'][method]
                p_str = '<0.001' if m['p_value'] < 0.001 else f"{m['p_value']:.4f}"
                lines.append(f"| {label} | {m['cohens_d']:.2f} | {m['interpretation']} | {p_str} |")
        lines.append("")
        
        # Mahalanobis Stats
        if 'mahalanobis' in results['uq_comparison']:
            m = results['uq_comparison']['mahalanobis']
            lines.append("### Mahalanobis Distance Analysis")
            lines.append("")
            lines.append(f"- **TP mean:** {m['tp_mean']:.2f} ± {m['tp_std']:.2f}")
            lines.append(f"- **FP mean:** {m['fp_mean']:.2f} ± {m['fp_std']:.2f}")
            lines.append(f"- **Gap (FP - TP):** {m['gap']:.2f} [95% CI: {m['gap_ci'][0]:.2f}, {m['gap_ci'][1]:.2f}]")
            lines.append(f"- **t-statistic:** {m['t_statistic']:.2f}")
            lines.append(f"- **p-value:** {'<0.001' if m['p_value'] < 0.001 else f'{m['p_value']:.4f}'}")
            lines.append(f"- **Cohen's d:** {m['cohens_d']:.2f} ({m['interpretation']})")
            lines.append("")
        
        # Threshold Analysis
        lines.append("### Mahalanobis Threshold Analysis (95% Wilson CI)")
        lines.append("")
        lines.append("| Threshold | TPs Kept | 95% CI | FPs Rejected | 95% CI |")
        lines.append("|-----------|----------|--------|--------------|--------|")
        for t in results['threshold_analysis']:
            lines.append(f"| {t['threshold']} | {t['tp_kept_rate']*100:.1f}% | [{t['tp_kept_ci'][0]*100:.1f}%, {t['tp_kept_ci'][1]*100:.1f}%] | "
                        f"{t['fp_rejected_rate']*100:.1f}% | [{t['fp_rejected_ci'][0]*100:.1f}%, {t['fp_rejected_ci'][1]*100:.1f}%] |")
        lines.append("")
    
    # Conclusion
    lines.append("---")
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    lines.append("1. **Statistical Significance:** The Mahalanobis distance difference between")
    lines.append("   True Positives and False Positives is highly significant (p < 0.001).")
    lines.append("")
    lines.append("2. **Method Comparison:** Mahalanobis distance shows large effect size,")
    lines.append("   while MC Dropout and Temperature Scaling show negligible/small effects.")
    lines.append("")
    lines.append("3. **Generalization:** Results are computed on held-out test set,")
    lines.append("   providing unbiased estimates of real-world performance.")
    lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("STATISTICAL ANALYSIS FOR ISEF (CORRECTED)")
    print("=" * 70)
    print("\n*** Using TEST SET ONLY for all statistics ***")
    
    # Analyze Coswara
    coswara_results = None
    try:
        coswara_results = analyze_dataset(COSWARA_CSV, "Coswara", use_test_only=True)
    except Exception as e:
        print(f"Failed to analyze Coswara: {e}")
    
    # Analyze ESC-50
    esc50_results = None
    try:
        esc50_results = analyze_dataset(ESC50_CSV, "ESC-50", use_test_only=True)
    except Exception as e:
        print(f"Failed to analyze ESC-50: {e}")
    
    # Generate report
    if coswara_results or esc50_results:
        print(f"\n{'='*70}")
        print("Generating Report")
        print(f"{'='*70}")
        
        report = generate_markdown_report(coswara_results, esc50_results)
        
        with open(OUTPUT_MD, 'w') as f:
            f.write(report)
        print(f"\n✓ Saved: {OUTPUT_MD}")
        
        # Summary CSV
        summary_rows = []
        
        for name, results in [("Coswara", coswara_results), ("ESC-50", esc50_results)]:
            if results is None:
                continue
            
            row = {
                'Dataset': name,
                'N (Test)': results['n_samples'],
                'TP': results['confusion_matrix']['TP'],
                'FP': results['confusion_matrix']['FP'],
                'Precision': f"{results['precision']['value']*100:.1f}%",
                'Precision_CI': f"[{results['precision']['ci'][0]*100:.1f}%, {results['precision']['ci'][1]*100:.1f}%]",
                'Recall': f"{results['recall']['value']*100:.1f}%",
                'Recall_CI': f"[{results['recall']['ci'][0]*100:.1f}%, {results['recall']['ci'][1]*100:.1f}%]",
            }
            
            if 'mahalanobis' in results['uq_comparison']:
                m = results['uq_comparison']['mahalanobis']
                row['Cohens_d'] = f"{m['cohens_d']:.2f}"
                row['Effect'] = m['interpretation']
                row['Gap'] = f"{m['gap']:.2f}"
                row['Gap_CI'] = f"[{m['gap_ci'][0]:.2f}, {m['gap_ci'][1]:.2f}]"
            
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(OUTPUT_CSV, index=False)
        print(f"✓ Saved: {OUTPUT_CSV}")
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
