from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.common.config_loader import ensure_dir, save_json


def _finalize(fig, output_path):
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_stage_metrics(result, output_dir, plot_format):
    stages = ['train_metrics', 'validation_metrics', 'test_metrics']
    metric_names = ['accuracy', 'auc', 'balanced_accuracy', 'f1']
    values = []
    for stage_name in stages:
        values.append([result[stage_name][metric] for metric in metric_names])
    values = np.asarray(values)
    x_axis = np.arange(len(metric_names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#123A8C', '#2E8B57', '#C24642']
    labels = ['Train', 'Validation', 'Test']
    for idx in range(3):
        ax.bar(x_axis + (idx - 1) * width, values[idx], width=width, color=colors[idx], label=labels[idx])
    ax.set_xticks(x_axis)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0.0, 1.05)
    ax.set_title('Stage Metrics')
    ax.legend()
    ax.grid(axis='y', alpha=0.2)
    _finalize(fig, Path(output_dir) / ('stage_metrics.' + plot_format))


def plot_roc_curve(result, output_dir, plot_format):
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr = np.asarray(result['test_metrics']['fpr'])
    tpr = np.asarray(result['test_metrics']['tpr'])
    auc_value = result['test_metrics']['auc']
    ax.plot(fpr, tpr, color='#C24642', label='AUC=%.3f' % auc_value)
    ax.plot([0, 1], [0, 1], '--', color='#666666')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(alpha=0.2)
    _finalize(fig, Path(output_dir) / ('roc_curve.' + plot_format))


def plot_confusion_matrix(result, output_dir, plot_format):
    matrix = np.asarray(result['test_metrics']['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap='Blues')
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha='center', va='center')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['rest', 'task'])
    ax.set_yticklabels(['rest', 'task'])
    ax.set_title('Confusion Matrix')
    fig.colorbar(image, ax=ax, fraction=0.046)
    _finalize(fig, Path(output_dir) / ('confusion_matrix.' + plot_format))


def plot_holdout_overview(summary, metric_name, output_dir, plot_format):
    labels = [fold['fold_name'] for fold in summary['folds']]
    values = [fold['best_result']['test_metrics'][metric_name] for fold in summary['folds']]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.3), 4))
    ax.bar(np.arange(len(labels)), values, color='#123A8C')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0.0, 1.05)
    ax.set_title('Holdout %s' % metric_name)
    ax.grid(axis='y', alpha=0.2)
    _finalize(fig, Path(output_dir) / ('holdout_%s.' % metric_name + plot_format))


def write_result_package(result, output_dir, report_config):
    output_dir = ensure_dir(output_dir)
    if report_config['save_json']:
        save_json(Path(output_dir) / 'metrics_summary.json', result)
    if report_config['save_plots']:
        plot_stage_metrics(result, output_dir, report_config['plot_format'])
        plot_roc_curve(result, output_dir, report_config['plot_format'])
        plot_confusion_matrix(result, output_dir, report_config['plot_format'])
