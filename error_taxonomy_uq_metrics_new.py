import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# 1. LOAD AND PREP THE DATA
# ==========================================
df_mahal = pd.read_csv("outputs/all_samples_T8p57_th0p54_coswara_20260318_145415_mahal_corrected.csv")
excel_file = "outputs/Cough Detection - Error Taxonomy Summary (1).xlsx"
df_tax = pd.read_excel(excel_file, sheet_name="UQ_Coswara")

# Clean column headers
df_tax.columns = df_tax.columns.str.strip()

# Extract IDs safely for merging
def extract_id_from_path(path):
    if pd.isnull(path):
        return "unknown"
    parts = str(path).replace('\\', '/').split('/')
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"

df_mahal['participant_id'] = df_mahal['file_path'].apply(extract_id_from_path)
df_tax['participant_id'] = df_tax['file_path'].apply(extract_id_from_path)

# Extract both FPs and FNs from manual taxonomy
df_tax_clean = df_tax[['filename', 'participant_id', 'Category Code']].dropna(subset=['Category Code'])

# ==========================================
# 2. SEPARATE AND COMBINE
# ==========================================
df_tp = df_mahal[df_mahal['result'] == 'TP'].copy()
df_tp['Category Code'] = 'True Cough (TP)'

df_fp = df_mahal[df_mahal['result'] == 'FP'].copy()
df_fp_merged = pd.merge(df_fp, df_tax_clean, on=['filename', 'participant_id'], how='inner')

df_fn = df_mahal[df_mahal['result'] == 'FN'].copy()
df_fn_merged = pd.merge(df_fn, df_tax_clean, on=['filename', 'participant_id'], how='inner')

df_plot = pd.concat([df_tp, df_fp_merged, df_fn_merged], ignore_index=True)

# ---> EXPLICIT CATEGORY MAPPING <---
def map_narrative_category(cat):
    if 'TP' in cat: 
        return '1. True Cough (TP)'
    elif 'FN-Regular' in cat:
        return '2. Missed Regular Cough (FN)'
    elif 'FN-Background' in cat:
        return '3. Missed Due to Noise (FN)'
    elif 'Breathing' in cat: 
        return '4. Breathing (FP)'
    elif 'Saying letter' in cat: 
        return '5. Saying Letter (FP)'
    elif 'Counting' in cat:
        return '6. Counting (FP)'
    else: 
        return '7. Other'

df_plot['Taxonomy Category'] = df_plot['Category Code'].apply(map_narrative_category)

# Filter explicitly for the top requested categories
df_plot_filtered = df_plot[df_plot['Taxonomy Category'].isin([
    '1. True Cough (TP)', 
    '2. Missed Regular Cough (FN)', 
    '3. Missed Due to Noise (FN)', 
    '4. Breathing (FP)', 
    '5. Saying Letter (FP)',
    '6. Counting (FP)'
])].sort_values(by='Taxonomy Category')

# ==========================================
# 3. GENERATE THE DASHBOARD
# ==========================================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharey=True)

# Colors: Green for TP, Purples for FNs, Orange/Reds for FPs
custom_palette = {
    '1. True Cough (TP)': '#2ecc71',         
    '2. Missed Regular Cough (FN)': '#9b59b6',  
    '3. Missed Due to Noise (FN)': '#8e44ad', 
    '4. Breathing (FP)': '#e67e22',
    '5. Saying Letter (FP)': '#e74c3c',
    '6. Counting (FP)': '#c0392b'
}

# Panel 1 (Leftmost): MC Dropout
sns.boxplot(data=df_plot_filtered, y='Taxonomy Category', x='epistemic_uncertainty', ax=axes[0], palette=custom_palette)
axes[0].set_title('Model Disagreement\n(MC Dropout Variance)', fontweight='bold')
axes[0].set_xlabel('Variance', fontweight='bold')
axes[0].set_ylabel('Taxonomy Category', fontweight='bold')

# Panel 2 (Middle): Temperature Scaling
sns.boxplot(data=df_plot_filtered, y='Taxonomy Category', x='predictive_entropy', ax=axes[1], palette=custom_palette)
axes[1].set_title('Total Uncertainty\n(Predictive Entropy via Temp Scaling)', fontweight='bold')
axes[1].set_xlabel('Entropy', fontweight='bold')
axes[1].set_ylabel('')

# Panel 3 (Rightmost): Mahalanobis Distance
sns.boxplot(data=df_plot_filtered, y='Taxonomy Category', x='mahalanobis_distance', ax=axes[2], palette=custom_palette)
axes[2].set_title('Feature Anomaly\n(Mahalanobis Distance)', fontweight='bold')
axes[2].set_xlabel('Distance Score', fontweight='bold')
axes[2].set_ylabel('') 
axes[2].axvline(x=35.25, color='black', linestyle='--', label='Threshold')

# ---> CLIPPING THE EXTREME OUTLIERS TO UN-SQUASH THE BOXES <---
# Clip MC Dropout at ~0.015 to ignore the extreme 0.23 outlier (Now on axes[0])
axes[0].set_xlim(left=-0.0005, right=0.015)
# Entropy doesn't usually have massive outliers, so we let it auto-scale (Now on axes[1])
# Clip Mahalanobis at ~85 to ignore the 170+ outlier (Now on axes[2])
axes[2].set_xlim(left=10, right=85)

plt.suptitle(
'The Diagnostic Matrix: Mapping Multi-Signal Uncertainty Quantification Methods to the Error Taxonomy',
             fontweight='bold', fontsize=18, y=1.05)
plt.tight_layout()
plt.savefig('Final_Zoomed_Dashboard1.png', dpi=300, bbox_inches='tight')

print("Final Dashboard generated successfully with correct order and bold labels!")
