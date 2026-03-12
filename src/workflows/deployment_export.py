from pathlib import Path

import joblib

from src.common.config_loader import ensure_dir, load_json, resolve_path, save_json
from src.evaluate.metrics import flatten_bundle
from src.models.trainer import evaluate_predictions, fit_fixed_model, save_model, select_threshold
from src.workflows.offline_runner import build_feature_inventory
from src.data.loader import merge_feature_bundles


def _sorted_day_keys(feature_bundles_by_day):
    return sorted(feature_bundles_by_day.keys(), key=lambda value: int(value.replace('day', '')))


def _collect_training_bundles(feature_bundles_by_day, train_days=None):
    day_keys = train_days or _sorted_day_keys(feature_bundles_by_day)
    bundles = []
    for day_key in day_keys:
        bundles.extend(feature_bundles_by_day[day_key])
    return merge_feature_bundles(bundles, metadata={'deployment_train_days': list(day_keys)})


def build_deployment_bundle(config_path):
    config = load_json(config_path)
    deployment_cfg = config['deployment']
    feature_bundles_by_day = build_feature_inventory(config)
    train_days = deployment_cfg.get('train_days')
    train_bundle = _collect_training_bundles(feature_bundles_by_day, train_days=train_days)
    x_train, y_train = flatten_bundle(train_bundle)

    fixed_training = fit_fixed_model(
        x_train,
        y_train,
        deployment_cfg['model_name'],
        config['model'],
        deployment_cfg.get('best_params')
    )

    if deployment_cfg.get('threshold_mode', 'fixed') == 'fixed':
        threshold = float(deployment_cfg['threshold'])
        threshold_info = {
            'mode': 'fixed',
            'selected_threshold': threshold
        }
    else:
        threshold, threshold_info = select_threshold(y_train, fixed_training['oof_scores'], config['model'])
        threshold_info['mode'] = 'oof_selected'

    deployment_metrics = evaluate_predictions(y_train, fixed_training['train_scores'], threshold)
    output_dir = ensure_dir(deployment_cfg['output_dir'])
    model_filename = deployment_cfg.get('model_filename', '%s_decoder_bundle.pkl' % config['task_type'])
    summary_filename = deployment_cfg.get('summary_filename', '%s_decoder_summary.json' % config['task_type'])

    bundle = {
        'bundle_type': 'deployment_decoder',
        'subject': config.get('subject'),
        'task_type': config.get('task_type'),
        'label_mapping': dict(train_bundle['label_mapping']),
        'train_days': list(train_bundle['metadata'].get('deployment_train_days', [])),
        'n_channels': config['n_channels'],
        'channel_select_mode': config.get('channel_select_mode', 'last'),
        'preprocess': dict(config['preprocess']),
        'features': dict(config['features']),
        'model_name': deployment_cfg['model_name'],
        'best_params': dict(deployment_cfg.get('best_params', {})),
        'threshold': float(threshold),
        'threshold_info': threshold_info,
        'train_metrics': deployment_metrics,
        'model': fixed_training['model']
    }

    model_path = Path(output_dir) / model_filename
    summary_path = Path(output_dir) / summary_filename
    save_model(model_path, bundle)
    save_json(summary_path, {
        'subject': bundle['subject'],
        'task_type': bundle['task_type'],
        'train_days': bundle['train_days'],
        'model_name': bundle['model_name'],
        'best_params': bundle['best_params'],
        'threshold': bundle['threshold'],
        'threshold_info': bundle['threshold_info'],
        'train_metrics': bundle['train_metrics'],
        'output_model_path': str(resolve_path(model_path))
    })
    return {
        'model_path': str(resolve_path(model_path)),
        'summary_path': str(resolve_path(summary_path)),
        'threshold': bundle['threshold'],
        'train_metrics': deployment_metrics
    }


def build_all_deployment_bundles(config_paths):
    results = []
    for config_path in config_paths:
        results.append(build_deployment_bundle(config_path))
    return results
