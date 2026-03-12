import numpy as np


def split_within_bundle(feature_bundle, train_ratio, shuffle_within_class=False, seed=42):
    """
    单天内的训练测试划分
    :param feature_bundle:
    :param train_ratio:
    :param shuffle_within_class:
    :param seed:
    :return:
    """
    rng = np.random.default_rng(seed)
    train_bundle = {
        'datasets': {},
        'label_mapping': dict(feature_bundle['label_mapping']),
        'fs': feature_bundle['fs'],
        'metadata': dict(feature_bundle.get('metadata', {}))
    }
    test_bundle = {
        'datasets': {},
        'label_mapping': dict(feature_bundle['label_mapping']),
        'fs': feature_bundle['fs'],
        'metadata': dict(feature_bundle.get('metadata', {}))
    }

    for class_id in sorted(feature_bundle['datasets']):
        data_array = np.asarray(feature_bundle['datasets'][class_id])
        indices = np.arange(data_array.shape[0])
        if shuffle_within_class:
            indices = rng.permutation(indices)
        split_index = int(len(indices) * train_ratio)
        split_index = max(1, min(split_index, len(indices) - 1))
        train_idx = indices[:split_index]
        test_idx = indices[split_index:]
        train_bundle['datasets'][class_id] = data_array[train_idx]
        test_bundle['datasets'][class_id] = data_array[test_idx]

    return train_bundle, test_bundle


def chronological_folds(grouped_recordings):
    """
    跨天的训练测试划分
    :param grouped_recordings:
    :return:
    """
    day_keys = sorted(grouped_recordings.keys(), key=lambda value: int(value.replace('day', '')))
    folds = []
    for idx in range(1, len(day_keys)):
        folds.append({
            'name': 'train_%s_test_%s' % ('_'.join(day_keys[:idx]), day_keys[idx]),
            'train_keys': day_keys[:idx],
            'test_keys': [day_keys[idx]]
        })
    return folds


def leave_one_day_out_folds(grouped_recordings):
    """
    :param grouped_recordings:
    :return:
    """
    day_keys = sorted(grouped_recordings.keys(), key=lambda value: int(value.replace('day', '')))
    folds = []
    for test_key in day_keys:
        train_keys = [key for key in day_keys if key != test_key]
        folds.append({
            'name': 'holdout_%s' % test_key,
            'train_keys': train_keys,
            'test_keys': [test_key]
        })
    return folds
