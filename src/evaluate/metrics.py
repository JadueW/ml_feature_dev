import numpy as np


def flatten_bundle(bundle):
    x_parts = []
    y_parts = []
    for class_id in sorted(bundle['datasets']):
        values = np.asarray(bundle['datasets'][class_id])
        x_parts.append(values)
        y_parts.append(np.full(values.shape[0], int(class_id), dtype=int))
    return np.vstack(x_parts), np.concatenate(y_parts)


def summarize_result(result):
    return {
        'model_name': result['model_name'],
        'best_params': result['best_params'],
        'selected_threshold': result['selected_threshold'],
        'validation_metrics': result['validation_metrics'],
        'test_metrics': result['test_metrics']
    }
