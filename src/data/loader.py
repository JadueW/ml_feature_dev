import joblib
import numpy as np


def load_raw_recording(path_value):
    """
    读取raw.pkl文件
    :param path_value:
    :return:
    """
    return joblib.load(path_value)


def select_channels(data_array, n_channels, mode='last'):
    """
    选择有效通道
    :param data_array:
    :param n_channels:
    :param mode:
    :return:
    """
    if mode == 'last':
        return data_array[:, -n_channels:, :]
    return data_array[:, :n_channels, :]


def build_binary_bundle(raw_payload, config, recording_info=None):
    datasets = raw_payload['datasets']
    rest_label = config['rest_label'] # 从配置文件中获取静息态的label
    task_labels = config['task_labels'] # 从配置文件中获取任务态的label
    n_channels = config['n_channels']
    channel_mode = config.get('channel_select_mode', 'last')

    if hasattr(datasets, 'items'):
        items = list(datasets.items())
    else:
        items = list(enumerate(datasets))

    normalized = {}
    for class_id, values in items:
        array = np.asarray(values)
        if array.ndim != 3:
            raise ValueError('Expected 3D array for class %s, got %s' % (class_id, array.shape))
        normalized[int(class_id)] = select_channels(array, n_channels, channel_mode)

    if rest_label not in normalized:
        raise KeyError('Rest label %s not found in datasets' % rest_label)

    rest_data = normalized[rest_label] # 把 datasets[0] 保留为 rest
    task_parts = [] #把其他任务类合并为 task
    for class_id in sorted(normalized):
        if class_id == rest_label:
            continue
        if task_labels and class_id not in task_labels:
            continue
        task_parts.append(normalized[class_id])

    if not task_parts:
        raise ValueError('No task segments found')

    task_data = np.concatenate(task_parts, axis=0)

    bundle = {
        'datasets': {0: rest_data, 1: task_data},
        'label_mapping': {0: 'rest', 1: 'task'},
        'fs': float(raw_payload['fs']),
        'metadata': {
            'recording_info': recording_info or {},
            'source': 'binary_rest_task',
            'n_channels': n_channels
        }
    }
    return bundle


def merge_feature_bundles(bundle_list, metadata=None):
    """
    #把多个 recording 的特征 bundle 合并
    :param bundle_list:
    :param metadata:
    :return:
    """
    if not bundle_list:
        raise ValueError('bundle_list is empty')
    merged = {
        'datasets': {},
        'label_mapping': dict(bundle_list[0]['label_mapping']),
        'fs': bundle_list[0]['fs'],
        'metadata': dict(bundle_list[0].get('metadata', {}))
    }
    if metadata:
        merged['metadata'].update(metadata)
    for class_id in sorted(bundle_list[0]['datasets']):
        merged['datasets'][class_id] = np.vstack([np.asarray(bundle['datasets'][class_id]) for bundle in bundle_list])
    return merged
