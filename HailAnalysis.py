"""
Author: Dara Procell
Date: December 12, 2025
Summary: Using various machine learning techniques to determine hail prediction.
Usage: python HailAnalysis.py [fixed_time|daily_max_cape]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             roc_auc_score, roc_curve, precision_recall_curve, 
                             auc, confusion_matrix)
from sklearn import __version__ as sklearn_version
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import learning_curve
from sklearn.tree import plot_tree
import os
import sys


# =============================================================================
# DATASET SELECTION
# =============================================================================

# Check if argument provided
if len(sys.argv) > 1:
    dataset_choice = sys.argv[1].lower()
    if dataset_choice not in ['fixed_time', 'daily_max_cape']:
        print(f"ERROR: Invalid argument '{sys.argv[1]}'")
        print("Usage: python HailAnalysis.py [fixed_time|daily_max_cape]")
        sys.exit(1)
else:
    print("Which dataset would you like to analyze?")
    print("  1. fixed_time      - 00Z fixed time extraction (Dataset 1)")
    print("  2. daily_max_cape  - Daily maximum CAPE extraction (Dataset 2)")
    
    while True:
        user_input = input("Enter your choice (1 or 2, or 'fixed_time'/'daily_max_cape'): ").strip().lower()
        
        if user_input in ['1', 'fixed_time']:
            dataset_choice = 'fixed_time'
            break
        elif user_input in ['2', 'daily_max_cape']:
            dataset_choice = 'daily_max_cape'
            break
        else:
            print("Invalid input. Please enter 1, 2, 'fixed_time', or 'daily_max_cape'")

if dataset_choice == 'fixed_time':
    data_file = 'data/final_dataset_fixed_time.csv'
    figures_dir = 'figures/fixed_time'  
    dataset_label = "00Z Fixed Time Extraction"
elif dataset_choice == 'daily_max_cape':
    data_file = 'data/final_dataset_daily_max_cape.csv'
    figures_dir = 'figures/daily_max_cape'  
    dataset_label = "Daily Maximum CAPE Extraction"

# Create output directory
os.makedirs(figures_dir, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# =============================================================================
# SECTION 1: LOAD DATA
# =============================================================================

df = pd.read_csv(data_file)
df['date'] = pd.to_datetime(df['date'])

print(df.shape)
print(df.describe())

# =============================================================================
# SECTION 2: TRAIN/TEST SPLIT
# =============================================================================

df = df.sort_values('date').reset_index(drop=True)

split_idx = int(len(df) * 0.7)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

# =============================================================================
# SECTION 3: FEATURE STANDARDIZATION
# =============================================================================

# features
features_full = ['cape', 'shear', 'freezing_level']
features_reduced = ['cape']

# training data
X_train_full = train_df[features_full]
X_train_reduced = train_df[features_reduced]
y_train = train_df['hail_occurred']

# test data
X_test_full = test_df[features_full]
X_test_reduced = test_df[features_reduced]
y_test = test_df['hail_occurred']

# Standardize features (z score normalization)
scaler_full = StandardScaler()
scaler_reduced = StandardScaler()

X_train_full_scaled = scaler_full.fit_transform(X_train_full)
X_test_full_scaled = scaler_full.transform(X_test_full)

X_train_reduced_scaled = scaler_reduced.fit_transform(X_train_reduced)
X_test_reduced_scaled = scaler_reduced.transform(X_test_reduced)

print("Original feature statistics (training set):")
print(train_df[features_full].describe())

print("Standardized feature statistics (training set):")
print(pd.DataFrame(X_train_full_scaled, columns=features_full).describe())

# =============================================================================
# SECTION 4: MODEL A & B - LINEAR LOGISTIC REGRESSION
# =============================================================================

def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """Calculate and display all evaluation metrics"""
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    # AUC scores
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall_vals, precision_vals)
    
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("ROC-AUC:", roc_auc)
    print("PR-AUC:", pr_auc)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': acc, 
        'precision': prec, 
        'recall': rec,
        'roc_auc': roc_auc, 
        'pr_auc': pr_auc,
        'cm': cm
    }


# Fit Model A (Full) with standardized features
model_full_scaled = LogisticRegression(random_state=42, max_iter=1000)
model_full_scaled.fit(X_train_full_scaled, y_train)

# Fit Model B (Reduced) with standardized features
model_reduced_scaled = LogisticRegression(random_state=42, max_iter=1000)
model_reduced_scaled.fit(X_train_reduced_scaled, y_train)

y_pred_full_scaled = model_full_scaled.predict(X_test_full_scaled)
y_pred_reduced_scaled = model_reduced_scaled.predict(X_test_reduced_scaled)

y_pred_proba_full_scaled = model_full_scaled.predict_proba(X_test_full_scaled)[:, 1]
y_pred_proba_reduced_scaled = model_reduced_scaled.predict_proba(X_test_reduced_scaled)[:, 1]

metrics_full_scaled = evaluate_model(y_test, y_pred_full_scaled, y_pred_proba_full_scaled, 
                                      "Model A (Full, Standardized)")
metrics_reduced_scaled = evaluate_model(y_test, y_pred_reduced_scaled, y_pred_proba_reduced_scaled, 
                                         "Model B (Reduced, Standardized)")

# =============================================================================
# SECTION 5: MODEL C - PRINCIPAL COMPONENT ANALYSIS (PCA)
# =============================================================================

# Fit PCA on training data
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_full_scaled)
X_test_pca = pca.transform(X_test_full_scaled)

loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2', 'PC3'],
    index=features_full
)

# Fit logistic regression on PCA components
model_pca = LogisticRegression(random_state=42, max_iter=1000)
model_pca.fit(X_train_pca, y_train)

y_pred_pca = model_pca.predict(X_test_pca)
y_pred_proba_pca = model_pca.predict_proba(X_test_pca)[:, 1]

metrics_pca = evaluate_model(y_test, y_pred_pca, y_pred_proba_pca, 
                              "Model C (PCA)")

# =============================================================================
# SECTION 6: MODEL D - RANDOM FOREST
# =============================================================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_full, y_train)

# feature importances
importances_df = pd.DataFrame({
    'feature': features_full,
    'importance': rf_model.feature_importances_,
    'std': np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
}).sort_values('importance', ascending=False)

y_pred_rf = rf_model.predict(X_test_full)
y_pred_proba_rf = rf_model.predict_proba(X_test_full)[:, 1]

metrics_rf = evaluate_model(y_test, y_pred_rf, y_pred_proba_rf, 
                             "Model D (Random Forest)")


# =============================================================================
# SECTION 6.5: MODEL E - CAPE + FREEZING LEVEL (NO SHEAR)
# =============================================================================

'''Hypothesis: Removing shear (weak correlation: +0.010) may improve performance
by reducing multicollinearity while keeping more meaningful variables'''

features_no_shear = ['cape', 'freezing_level']

X_train_no_shear = train_df[features_no_shear]
X_test_no_shear = test_df[features_no_shear]

# Standardize
scaler_no_shear = StandardScaler()
X_train_no_shear_scaled = scaler_no_shear.fit_transform(X_train_no_shear)
X_test_no_shear_scaled = scaler_no_shear.transform(X_test_no_shear)

# Fit Model E
model_no_shear = LogisticRegression(random_state=42, max_iter=1000)
model_no_shear.fit(X_train_no_shear_scaled, y_train)

y_pred_no_shear = model_no_shear.predict(X_test_no_shear_scaled)
y_pred_proba_no_shear = model_no_shear.predict_proba(X_test_no_shear_scaled)[:, 1]

metrics_no_shear = evaluate_model(y_test, y_pred_no_shear, y_pred_proba_no_shear, 
                                   "Model E (CAPE + Freezing Level, No Shear)")

# =============================================================================
# SECTION 7: MODEL COMPARISON (ALL 5 MODELS)
# =============================================================================

# Create comparison dataframe, excluding confusion matrix
comparison_all = pd.DataFrame({
    'Model A (Linear Full)': {k: v for k, v in metrics_full_scaled.items() if k != 'cm'},
    'Model B (Linear CAPE)': {k: v for k, v in metrics_reduced_scaled.items() if k != 'cm'},
    'Model C (PCA)': {k: v for k, v in metrics_pca.items() if k != 'cm'},
    'Model D (Random Forest)': {k: v for k, v in metrics_rf.items() if k != 'cm'},
    'Model E (CAPE + Frz Lvl)': {k: v for k, v in metrics_no_shear.items() if k != 'cm'}
})

print(comparison_all.round(3))

# Highlight best model for each metric
print("Best model for each metric:")
for metric in comparison_all.index:
    best_model = comparison_all.loc[metric].idxmax()
    best_score = comparison_all.loc[metric].max()
    print(f"  {metric:15s}: {best_model} ({best_score:.3f})")


# =============================================================================
# SECTION 8: VISUALIZATIONS
# =============================================================================

# Figure 1: ROC and PR Curves (Original Models A & B)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC Curve
fpr_full, tpr_full, _ = roc_curve(y_test, y_pred_proba_full_scaled)
fpr_reduced, tpr_reduced, _ = roc_curve(y_test, y_pred_proba_reduced_scaled)

axes[0].plot(fpr_full, tpr_full, label=f"Model A (AUC={metrics_full_scaled['roc_auc']:.3f})", 
             linewidth=2.5, color='#2E86AB')
axes[0].plot(fpr_reduced, tpr_reduced, label=f"Model B (AUC={metrics_reduced_scaled['roc_auc']:.3f})", 
             linewidth=2.5, color='#A23B72')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Precision-Recall Curve
prec_full, rec_full, _ = precision_recall_curve(y_test, y_pred_proba_full_scaled)
prec_reduced, rec_reduced, _ = precision_recall_curve(y_test, y_pred_proba_reduced_scaled)

axes[1].plot(rec_full, prec_full, label=f"Model A (AUC={metrics_full_scaled['pr_auc']:.3f})", 
             linewidth=2.5, color='#2E86AB')
axes[1].plot(rec_reduced, prec_reduced, label=f"Model B (AUC={metrics_reduced_scaled['pr_auc']:.3f})", 
             linewidth=2.5, color='#A23B72')
axes[1].axhline(y=y_test.mean(), color='k', linestyle='--', alpha=0.3, 
                label=f'Baseline ({y_test.mean():.3f})')
axes[1].set_xlabel('Recall', fontsize=12)
axes[1].set_ylabel('Precision', fontsize=12)
axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{figures_dir}/roc_pr_curves.png', dpi=300, bbox_inches='tight')
plt.close()


# Figure 2: Metrics Comparison (Models A & B)

fig, ax = plt.subplots(figsize=(10, 6))

metrics_to_plot = ['accuracy', 'precision', 'recall', 'roc_auc', 'pr_auc']
x = np.arange(len(metrics_to_plot))
width = 0.35

full_scores = [metrics_full_scaled[m] for m in metrics_to_plot]
reduced_scores = [metrics_reduced_scaled[m] for m in metrics_to_plot]

bars1 = ax.bar(x - width/2, full_scores, width, label='Model A (Full)', 
               color='#2E86AB', alpha=0.8)
bars2 = ax.bar(x + width/2, reduced_scores, width, label='Model B (Reduced)', 
               color='#A23B72', alpha=0.8)

ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC'])
ax.legend()
ax.set_ylim([0, 1.0])
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{figures_dir}/metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()


# Figure 3: Feature Distributions

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, feature in enumerate(features_full):
    axes[idx].hist(df[df['hail_occurred']==0][feature], bins=30, alpha=0.6, 
                   label='No Hail', color='#2E86AB', density=True)
    axes[idx].hist(df[df['hail_occurred']==1][feature], bins=30, alpha=0.6, 
                   label='Hail', color='#A23B72', density=True)
    axes[idx].set_xlabel(feature.replace('_', ' ').title(), fontsize=11)
    axes[idx].set_ylabel('Density', fontsize=11)
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

plt.suptitle('Feature Distributions by Hail Occurrence', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{figures_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()


# Figure 4: Confusion Matrices

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (metrics, title) in enumerate([(metrics_full_scaled, 'Model A (Full)'), 
                                          (metrics_reduced_scaled, 'Model B (Reduced)')]):
    cm = metrics['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                cbar_kws={'label': 'Count'})
    axes[idx].set_title(title, fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Actual', fontsize=11)
    axes[idx].set_xlabel('Predicted', fontsize=11)
    axes[idx].set_xticklabels(['No Hail', 'Hail'])
    axes[idx].set_yticklabels(['No Hail', 'Hail'])

plt.tight_layout()
plt.savefig(f'{figures_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()


# Figure 5: Coefficient Comparison (Original vs Standardized)

model_full_original = LogisticRegression(random_state=42, max_iter=1000)
model_full_original.fit(X_train_full, y_train)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original coefficients
coef_original = model_full_original.coef_[0]
axes[0].barh(features_full, coef_original, color='#2E86AB', alpha=0.8)
axes[0].set_xlabel('Coefficient Value', fontsize=12)
axes[0].set_title('Original Coefficients\n(Pre z score normalization)', fontsize=12, fontweight='bold')
axes[0].axvline(x=0, color='black', linestyle='--', alpha=0.3)
axes[0].grid(axis='x', alpha=0.3)

# Standardized coefficients
coef_standardized = model_full_scaled.coef_[0]
axes[1].barh(features_full, coef_standardized, color='#A23B72', alpha=0.8)
axes[1].set_xlabel('Standardized Coefficient Value', fontsize=12)
axes[1].set_title('Standardized Coefficients\n(Post z score normalization)', fontsize=12, fontweight='bold')
axes[1].axvline(x=0, color='black', linestyle='--', alpha=0.3)
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{figures_dir}/coefficient_comparison.png', dpi=300, bbox_inches='tight')
plt.close()


# Figure 6: PCA Loadings and Variance

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Heatmap of loadings
sns.heatmap(loadings, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            ax=axes[0], cbar_kws={'label': 'Loading'}, vmin=-1, vmax=1)
axes[0].set_title('PCA Component Loadings\n(How features relate to PCs)', 
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Original Feature')

# Right: Explained variance
pc_labels = [f'PC{i+1}\n({var*100:.1f}%)' 
             for i, var in enumerate(pca.explained_variance_ratio_)]
axes[1].bar(range(3), pca.explained_variance_ratio_, 
            color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8)
axes[1].set_xlabel('Principal Component', fontsize=12)
axes[1].set_ylabel('Explained Variance Ratio', fontsize=12)
axes[1].set_title('Variance Explained by Each Component', 
                  fontsize=12, fontweight='bold')
axes[1].set_xticks(range(3))
axes[1].set_xticklabels(pc_labels)
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_ylim([0, 1])
cumsum = pca.explained_variance_ratio_.cumsum()
ax2 = axes[1].twinx()
ax2.plot(range(3), cumsum, 'ro-', linewidth=2, markersize=8, label='Cumulative')
ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
ax2.set_ylim([0, 1.1])
ax2.legend(loc='lower right')

plt.tight_layout()
plt.savefig(f'{figures_dir}/pca_loadings_variance.png', dpi=300, bbox_inches='tight')
plt.close()


# Figure 7: PCA Biplot

fig, ax = plt.subplots(figsize=(10, 8))

all_data_scaled = scaler_full.transform(df[features_full])
all_data_pca = pca.transform(all_data_scaled)

# Scatter plot
scatter = ax.scatter(all_data_pca[df['hail_occurred']==0, 0], 
                     all_data_pca[df['hail_occurred']==0, 1],
                     c='#2E86AB', alpha=0.4, s=30, label='No Hail')
scatter = ax.scatter(all_data_pca[df['hail_occurred']==1, 0], 
                     all_data_pca[df['hail_occurred']==1, 1],
                     c='#A23B72', alpha=0.4, s=30, label='Hail')

scale_factor = 4
for i, feature in enumerate(features_full):
    ax.arrow(0, 0, 
             loadings.iloc[i, 0] * scale_factor, 
             loadings.iloc[i, 1] * scale_factor,
             head_width=0.2, head_length=0.2, fc='red', ec='red', linewidth=2)
    ax.text(loadings.iloc[i, 0] * scale_factor * 1.15, 
            loadings.iloc[i, 1] * scale_factor * 1.15,
            feature.replace('_', ' ').title(), 
            fontsize=12, fontweight='bold', color='red',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
ax.set_title('PCA Biplot: Data Points and Feature Vectors', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{figures_dir}/pca_biplot.png', dpi=300, bbox_inches='tight')
plt.close()


# Figure 8: Random Forest Feature Importances

fig, ax = plt.subplots(figsize=(10, 6))

importances_df_sorted = importances_df.sort_values('importance', ascending=True)

bars = ax.barh(importances_df_sorted['feature'], 
               importances_df_sorted['importance'],
               xerr=importances_df_sorted['std'],
               color='#2E86AB', alpha=0.8, capsize=5)

ax.set_xlabel('Feature Importance (Mean Decrease in Gini Impurity)', fontsize=12)
ax.set_title('Random Forest Feature Importances', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

for idx, (bar, row) in enumerate(zip(bars, importances_df_sorted.iterrows())):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2, 
            f' {width:.3f}',
            va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{figures_dir}/rf_feature_importances.png', dpi=300, bbox_inches='tight')
plt.close()


# Figure 9: Partial Dependence Plots
    
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, (feature_idx, feature_name) in enumerate(zip([0, 1, 2], features_full)):
    pd_result = partial_dependence(
        rf_model,
        X_train_full,
        features=[feature_idx],
        grid_resolution=50
    )

# Handle different sklearn versions. My home machine has an older version, work machine has newer.
    if isinstance(pd_result, dict):
        # New sklearn (≥1.0): returns dict with 'average' and 'grid_values'
        avg_preds = pd_result['average'][0]
        values = pd_result['grid_values'][0]
    else:
        # Old sklearn (<1.0): returns tuple
        avg_preds = pd_result[0][0]
        values = pd_result[1][0]

    # avg_preds = pd_result[0][0]
    # values = pd_result[1][0]

    axes[idx].plot(values, avg_preds, linewidth=2.5, color='#2E86AB')
    axes[idx].set_xlabel(feature_name.replace('_', ' ').title(), fontsize=11)
    axes[idx].set_ylabel('Partial Dependence', fontsize=11)
    axes[idx].set_title(feature_name.replace('_', ' ').title(), fontsize=12)
    axes[idx].grid(alpha=0.3)

plt.suptitle('Partial Dependence Plots: How Each Feature Affects Hail Probability', 
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(f'{figures_dir}/rf_partial_dependence.png', dpi=300, bbox_inches='tight')
plt.close()


# Figure 10: Sample Decision Tree

fig, ax = plt.subplots(figsize=(20, 10))

plot_tree(rf_model.estimators_[0], 
          feature_names=features_full,
          class_names=['No Hail', 'Hail'],
          filled=True,
          rounded=True,
          ax=ax,
          max_depth=3,
          fontsize=10)

ax.set_title('Sample Decision Tree from Random Forest (Depth Limited to 3)', 
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{figures_dir}/rf_sample_tree.png', dpi=300, bbox_inches='tight')
plt.close()


# Figure 11: All Models ROC Comparison 

fig, ax = plt.subplots(figsize=(10, 8))

models_roc = [
    ('Model A (Linear Full)', y_pred_proba_full_scaled, '#2E86AB'),
    ('Model B (Linear CAPE)', y_pred_proba_reduced_scaled, '#A23B72'),
    ('Model C (PCA)', y_pred_proba_pca, '#F18F01'),
    ('Model D (Random Forest)', y_pred_proba_rf, '#06A77D'),
    ('Model E (CAPE + Frz Lvl)', y_pred_proba_no_shear, '#C73E1D')
]

for name, y_proba, color in models_roc:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})", 
            linewidth=2.5, color=color)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves: All Models Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{figures_dir}/all_models_roc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()


# Figure 12: All Models Metrics Comparison 

fig, ax = plt.subplots(figsize=(16, 6))

metrics_to_plot = ['accuracy', 'precision', 'recall', 'roc_auc', 'pr_auc']
x = np.arange(len(metrics_to_plot))
width = 0.16  

full_scores = [metrics_full_scaled[m] for m in metrics_to_plot]
reduced_scores = [metrics_reduced_scaled[m] for m in metrics_to_plot]
pca_scores = [metrics_pca[m] for m in metrics_to_plot]
rf_scores = [metrics_rf[m] for m in metrics_to_plot]
no_shear_scores = [metrics_no_shear[m] for m in metrics_to_plot]

bars1 = ax.bar(x - 2*width, full_scores, width, label='Model A (Linear Full)', 
               color='#2E86AB', alpha=0.8)
bars2 = ax.bar(x - width, reduced_scores, width, label='Model B (Linear CAPE)', 
               color='#A23B72', alpha=0.8)
bars3 = ax.bar(x, pca_scores, width, label='Model C (PCA)', 
               color='#F18F01', alpha=0.8)
bars4 = ax.bar(x + width, rf_scores, width, label='Model D (Random Forest)', 
               color='#06A77D', alpha=0.8)
bars5 = ax.bar(x + 2*width, no_shear_scores, width, label='Model E (CAPE + Frz Lvl)', 
               color='#C73E1D', alpha=0.8)

ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Performance Comparison: All Five Models', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC'])
ax.legend(fontsize=9, loc='lower right')
ax.set_ylim([0, 1.0])
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2, bars3, bars4, bars5]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=6)

plt.tight_layout()
plt.savefig(f'{figures_dir}/all_models_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()


# Figure 13: Learning Curve

train_sizes, train_scores, val_scores = learning_curve(
    rf_model, 
    X_train_full, 
    y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(train_sizes, train_mean, 'o-', color='#2E86AB', label='Training Score', linewidth=2)
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                alpha=0.2, color='#2E86AB')

ax.plot(train_sizes, val_mean, 'o-', color='#A23B72', label='Cross-Validation Score', linewidth=2)
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                alpha=0.2, color='#A23B72')

ax.set_xlabel('Training Set Size', fontsize=12)
ax.set_ylabel('ROC-AUC Score', fontsize=12)
ax.set_title('Random Forest Learning Curve', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

gap = train_mean[-1] - val_mean[-1]
if gap > 0.1:
    ax.text(0.5, 0.05, f'Gap = {gap:.3f} overfitting', 
            transform=ax.transAxes, fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
else:
    ax.text(0.5, 0.05, f'Gap = {gap:.3f}', 
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{figures_dir}/rf_learning_curve.png', dpi=300, bbox_inches='tight')
plt.close()

