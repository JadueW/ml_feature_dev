from pathlib import Path

import joblib

from src.common.config_loader import ensure_dir, load_json, resolve_path, save_json
from src.data.indexer import discover_recordings, group_by_day
from src.data.loader import build_binary_bundle, load_raw_recording, merge_feature_bundles
from src.evaluate.metrics import flatten_bundle, summarize_result
from src.evaluate.reports import plot_holdout_overview, write_result_package
from src.features.psd_features import extract_feature_bundle
from src.models.trainer import choose_best_result, save_model, train_one_model
from src.preprocess.pipeline import preprocess_bundle
from src.split.strategies import chronological_folds, leave_one_day_out_folds, split_within_bundle


def get_recording_cache_path(recording_info, config):
    cache_root = ensure_dir(config['cache_dir'])
    return Path(cache_root) / (recording_info['path'].stem + '_features.pkl')


def build_feature_bundle_for_recording(recording_info, config):
    cache_path = get_recording_cache_path(recording_info, config)
    if cache_path.exists():
        return joblib.load(cache_path)

    raw_payload = load_raw_recording(recording_info['path'])
    bundle = build_binary_bundle(raw_payload, config, recording_info=recording_info)
    processed = preprocess_bundle(bundle, config['preprocess'])
    features = extract_feature_bundle(processed, config['features'])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(features, cache_path)
    return features


def train_model_suite(train_bundle, test_bundle, config, model_names=None):
    x_train, y_train = flatten_bundle(train_bundle)
    x_test, y_test = flatten_bundle(test_bundle)
    model_names = model_names or config['model']['candidates']
    results = []
    for model_name in model_names:
        result = train_one_model(x_train, y_train, x_test, y_test, model_name, config['model'])
        if result is not None:
            results.append(result)
    best_result = choose_best_result(results, config['model']['refit_metric'])
    return best_result, results


def run_single_day_experiments(feature_bundles_by_day, config):
    reports = []
    split_cfg = config['splits']['single_day']
    output_root = ensure_dir(Path(config['output_dir']) / 'single_day')
    for day_key in sorted(feature_bundles_by_day.keys(), key=lambda value: int(value.replace('day', ''))):
        merged_day_bundle = merge_feature_bundles(feature_bundles_by_day[day_key], metadata={'split_mode': 'single_day'})
        train_bundle, test_bundle = split_within_bundle(
            merged_day_bundle,
            split_cfg['train_ratio'],
            split_cfg['shuffle_within_class'],
            config['model']['random_state']
        )
        best_result, all_results = train_model_suite(train_bundle, test_bundle, config)
        report_payload = {
            'fold_name': day_key,
            'split_mode': 'single_day',
            'candidate_results': [summarize_result(item) for item in all_results],
            'best_result': best_result
        }
        fold_output = output_root / day_key
        write_result_package(report_payload['best_result'], fold_output, config['report'])
        save_json(fold_output / 'experiment_summary.json', report_payload)
        if config['report']['save_model']:
            save_model(fold_output / 'best_model.pkl', best_result['best_model'])
        reports.append(report_payload)
    return reports


def _collect_bundles_for_keys(feature_bundles_by_day, day_keys):
    collected = []
    for day_key in day_keys:
        collected.extend(feature_bundles_by_day[day_key])
    return collected


def summarize_fold_reports(reports, split_mode):
    summary = {
        'split_mode': split_mode,
        'n_folds': len(reports),
        'folds': []
    }
    for report in reports:
        summary['folds'].append({
            'fold_name': report['fold_name'],
            'best_result': summarize_result(report['best_result'])
        })
    if reports:
        for metric_name in ['auc', 'accuracy', 'balanced_accuracy', 'f1']:
            values = [report['best_result']['test_metrics'][metric_name] for report in reports]
            mean_value = sum(values) / len(values)
            std_value = (sum((value - mean_value) ** 2 for value in values) / len(values)) ** 0.5
            summary['test_%s_mean' % metric_name] = float(mean_value)
            summary['test_%s_std' % metric_name] = float(std_value)
    return summary


def run_cross_day_experiments(feature_bundles_by_day, config, fold_mode):
    output_root = ensure_dir(Path(config['output_dir']) / fold_mode)
    if fold_mode == 'leave_one_day_out':
        fold_defs = leave_one_day_out_folds(feature_bundles_by_day)
    else:
        fold_defs = chronological_folds(feature_bundles_by_day)

    reports = []
    for fold_def in fold_defs:
        train_bundle = merge_feature_bundles(
            _collect_bundles_for_keys(feature_bundles_by_day, fold_def['train_keys']),
            metadata={'split_mode': fold_mode, 'fold_name': fold_def['name']}
        )
        test_bundle = merge_feature_bundles(
            _collect_bundles_for_keys(feature_bundles_by_day, fold_def['test_keys']),
            metadata={'split_mode': fold_mode, 'fold_name': fold_def['name']}
        )
        best_result, all_results = train_model_suite(train_bundle, test_bundle, config)
        report_payload = {
            'fold_name': fold_def['name'],
            'split_mode': fold_mode,
            'train_days': list(fold_def['train_keys']),
            'test_days': list(fold_def['test_keys']),
            'candidate_results': [summarize_result(item) for item in all_results],
            'best_result': best_result
        }
        fold_output = output_root / fold_def['name']
        write_result_package(report_payload['best_result'], fold_output, config['report'])
        save_json(fold_output / 'experiment_summary.json', report_payload)
        if config['report']['save_model']:
            save_model(fold_output / 'best_model.pkl', best_result['best_model'])
        reports.append(report_payload)

    summary = summarize_fold_reports(reports, fold_mode)
    save_json(output_root / 'summary.json', summary)
    if config['report']['save_plots'] and reports:
        plot_holdout_overview(summary, 'auc', output_root, config['report']['plot_format'])
        plot_holdout_overview(summary, 'balanced_accuracy', output_root, config['report']['plot_format'])
    return summary


def build_feature_inventory(config):
    raw_dir = resolve_path(config['raw_data_dir'])
    recordings = discover_recordings(raw_dir, subject=config.get('subject'), task_type=config.get('task_type'))
    if not recordings:
        raise FileNotFoundError('No recordings matched current config filters')
    grouped = group_by_day(recordings)
    feature_bundles_by_day = {}
    for day_key in sorted(grouped.keys(), key=lambda value: int(value.replace('day', ''))):
        feature_bundles_by_day[day_key] = []
        for recording_info in grouped[day_key]:
            feature_bundles_by_day[day_key].append(build_feature_bundle_for_recording(recording_info, config))
    return feature_bundles_by_day


def run_offline_experiments(config_path='configs/baseline_v2.json'):
    config = load_json(config_path)
    output_root = ensure_dir(config['output_dir'])
    feature_bundles_by_day = build_feature_inventory(config)

    results = {
        'experiment_name': config['experiment_name'],
        'subject': config.get('subject'),
        'task_type': config.get('task_type')
    }

    if config['splits']['single_day']['enabled']:
        results['single_day'] = run_single_day_experiments(feature_bundles_by_day, config)
    if config['splits']['leave_one_day_out']['enabled']:
        results['leave_one_day_out'] = run_cross_day_experiments(feature_bundles_by_day, config, 'leave_one_day_out')
    if config['splits']['chronological']['enabled']:
        results['chronological'] = run_cross_day_experiments(feature_bundles_by_day, config, 'chronological')

    save_json(output_root / 'run_summary.json', results)
    return results
