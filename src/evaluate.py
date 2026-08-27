import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_key(metrics, candidates):
    for key in candidates:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
    return None


def plot_avg_roc(y_true, y_score, output_path):
    if y_true is None or y_score is None or len(y_true) == 0 or len(y_score) == 0:
        raise ValueError("Missing true labels or prediction probabilities for ROC curve.")

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]

    if len(np.unique(y_true)) < 2:
        raise ValueError("ROC curve requires both classes in true labels.")

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], '--', color='grey', lw=1.5, label='Random (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right', frameon=True)
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_biological_variance(per_subject_auroc, output_path):
    auroc_values = [v for v in per_subject_auroc if v is not None and np.isfinite(v)]
    if not auroc_values:
        raise ValueError("No valid per-subject AUROC scores available for biological variance plot.")

    plt.figure(figsize=(8, 5))
    sns.violinplot(
        y=auroc_values,
        inner='box',
        cut=0,
        color='#2ca02c',
        linewidth=1.2
    )
    plt.ylabel('Per-subject AUROC')
    plt.title('Distribution of Per-subject AUROC Scores')
    plt.axhline(np.mean(auroc_values), color='red', linestyle='--', lw=1.5, label=f'Mean = {np.mean(auroc_values):.3f}')
    plt.legend(loc='upper right')
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_path):
    if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Missing true labels or predicted labels for confusion matrix.")

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct = np.nan_to_num(cm / row_sums * 100)

    annot_text = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_text[i, j] = f'{cm[i, j]}\n({pct[i, j]:.1f}%)'

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        cm,
        annot=annot_text,
        fmt='',
        cmap='Blues',
        cbar=False,
        xticklabels=['Predicted 0', 'Predicted 1'],
        yticklabels=['True 0', 'True 1'],
        annot_kws={'size': 11},
        linewidths=1,
        linecolor='white'
    )
    ax.set_title('Confusion Matrix (Global)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained run and generate publication-ready plots.')
    parser.add_argument('--run', required=True, help='Run directory name inside results/ (e.g., run_001)')
    args = parser.parse_args()

    run_name = args.run
    run_dir = os.path.join(PROJECT_ROOT, 'results', run_name)

    if not os.path.isdir(run_dir):
        print(f"Error: run directory not found: {run_dir}")
        sys.exit(1)

    metrics_path = os.path.join(run_dir, 'metrics.json')
    if not os.path.isfile(metrics_path):
        print(f"Error: metrics.json not found in {run_dir}")
        sys.exit(1)

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    sns.set_theme(style='whitegrid')

    y_true = find_key(metrics, [
        'true_labels', 'global_true_labels', 'y_true', 'test_labels',
        'global_labels', 'test_true_labels'
    ])
    y_score = find_key(metrics, [
        'probabilities', 'global_probabilities', 'y_score', 'test_probabilities',
        'global_scores', 'test_scores'
    ])
    y_pred = find_key(metrics, [
        'predictions', 'global_predictions', 'y_pred', 'test_predictions',
        'global_preds', 'test_preds'
    ])

    if y_pred is None and y_score is not None:
        y_pred = (np.asarray(y_score) >= 0.5).astype(int)

    fold_metrics = metrics.get('fold_metrics', [])
    per_subject_auroc = []
    for fold in fold_metrics:
        if isinstance(fold, dict) and 'auroc' in fold:
            per_subject_auroc.append(fold['auroc'])

    missing_messages = []

    try:
        plot_avg_roc(y_true, y_score, os.path.join(plots_dir, 'average_roc.png'))
        print(f"Saved: {os.path.join(plots_dir, 'average_roc.png')}")
    except Exception as exc:
        missing_messages.append(f"Average ROC plot skipped: {exc}")

    try:
        plot_biological_variance(per_subject_auroc, os.path.join(plots_dir, 'biological_variance_auroc.png'))
        print(f"Saved: {os.path.join(plots_dir, 'biological_variance_auroc.png')}")
    except Exception as exc:
        missing_messages.append(f"Biological variance plot skipped: {exc}")

    try:
        plot_confusion_matrix(y_true, y_pred, os.path.join(plots_dir, 'confusion_matrix.png'))
        print(f"Saved: {os.path.join(plots_dir, 'confusion_matrix.png')}")
    except Exception as exc:
        missing_messages.append(f"Confusion matrix plot skipped: {exc}")

    if missing_messages:
        print('\nSome plots could not be generated:')
        for msg in missing_messages:
            print(f'- {msg}')


if __name__ == '__main__':
    main()
