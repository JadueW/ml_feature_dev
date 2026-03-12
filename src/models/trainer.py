import joblib
import warnings
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.models.model_factory import create_model_pipeline, create_param_grid


warnings.filterwarnings('ignore', category=FutureWarning)

SCORING = {
    'auc': 'roc_auc',
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'f1': 'f1'
}


def get_prediction_scores(estimator, x_data):
    if hasattr(estimator, 'predict_proba'):
        return estimator.predict_proba(x_data)[:, 1]
    if hasattr(estimator, 'decision_function'):
        scores = np.asarray(estimator.decision_function(x_data), dtype=float)
        score_min = scores.min()
        score_max = scores.max()
        if np.isclose(score_min, score_max):
            return np.zeros_like(scores) + 0.5
        return (scores - score_min) / (score_max - score_min)
    return np.asarray(estimator.predict(x_data), dtype=float)


def evaluate_predictions(y_true, scores, threshold):
    predicted = (scores >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, scores)
    return {
        'accuracy': float(accuracy_score(y_true, predicted)),
        'auc': float(roc_auc_score(y_true, scores)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, predicted)),
        'f1': float(f1_score(y_true, predicted)),
        'confusion_matrix': confusion_matrix(y_true, predicted).tolist(),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist()
    }


def score_threshold_metric(y_true, predicted, metric_name):
    if metric_name == 'accuracy':
        return float(accuracy_score(y_true, predicted))
    if metric_name == 'balanced_accuracy':
        return float(balanced_accuracy_score(y_true, predicted))
    if metric_name == 'f1':
        return float(f1_score(y_true, predicted))
    raise ValueError('Unsupported threshold metric: %s' % metric_name)


def compute_oof_scores(best_model, x_train, y_train, n_splits, random_state):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_scores = np.zeros(len(y_train), dtype=float)
    for fold_train, fold_valid in cv.split(x_train, y_train):
        model = clone(best_model)
        model.fit(x_train[fold_train], y_train[fold_train])
        oof_scores[fold_valid] = get_prediction_scores(model, x_train[fold_valid])
    return oof_scores


def select_threshold(y_true, scores, model_config):
    """
    训练集 OOF 阈值选择
    :param y_true:
    :param scores:
    :param model_config:
    :return:
    """
    metric_name = model_config['threshold_metric']
    threshold_grid = model_config['threshold_grid']
    best_threshold = 0.5
    best_score = -1.0
    score_table = []
    for threshold in threshold_grid:
        predicted = (scores >= threshold).astype(int)
        metric_value = score_threshold_metric(y_true, predicted, metric_name)
        score_table.append({'threshold': float(threshold), 'score': float(metric_value)})
        if metric_value > best_score:
            best_score = metric_value
            best_threshold = float(threshold)
    return best_threshold, {
        'metric': metric_name,
        'selected_threshold': best_threshold,
        'selected_score': best_score,
        'scores': score_table
    }


def summarize_cv(cv_results, best_index):
    summary = {}
    for metric_name in ['auc', 'accuracy', 'balanced_accuracy', 'f1']:
        summary[metric_name] = {
            'mean_train': np.asarray(cv_results['mean_train_%s' % metric_name], dtype=float).tolist(),
            'mean_validation': np.asarray(cv_results['mean_test_%s' % metric_name], dtype=float).tolist(),
            'std_train': np.asarray(cv_results['std_train_%s' % metric_name], dtype=float).tolist(),
            'std_validation': np.asarray(cv_results['std_test_%s' % metric_name], dtype=float).tolist(),
            'selected_index': int(best_index)
        }
    return summary


def train_one_model(x_train, y_train, x_test, y_test, model_name, model_config):
    """
    网格搜索
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param model_name:
    :param model_config:
    :return:
    """
    pipeline = create_model_pipeline(model_name, model_config)
    if pipeline is None:
        return None
    param_grid = create_param_grid(model_name, model_config)
    inner_cv = StratifiedKFold(
        n_splits=model_config['inner_cv_splits'],
        shuffle=True,
        random_state=model_config['random_state']
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=SCORING,
        refit=model_config['refit_metric'],
        cv=inner_cv,
        return_train_score=True,
        verbose=0
    )
    search.fit(x_train, y_train)

    best_model = search.best_estimator_
    train_scores = get_prediction_scores(best_model, x_train)
    test_scores = get_prediction_scores(best_model, x_test)
    oof_scores = compute_oof_scores(best_model, x_train, y_train, model_config['inner_cv_splits'], model_config['random_state'])
    selected_threshold, threshold_info = select_threshold(y_train, oof_scores, model_config)

    return {
        'model_name': model_name,
        'best_model': best_model,
        'best_params': search.best_params_,
        'best_score': float(search.best_score_),
        'selected_threshold': selected_threshold,
        'threshold_info': threshold_info,
        'train_metrics': evaluate_predictions(y_train, train_scores, selected_threshold),
        'validation_metrics': {
            'auc': float(search.cv_results_['mean_test_auc'][search.best_index_]),
            'accuracy': float(search.cv_results_['mean_test_accuracy'][search.best_index_]),
            'balanced_accuracy': float(search.cv_results_['mean_test_balanced_accuracy'][search.best_index_]),
            'f1': float(search.cv_results_['mean_test_f1'][search.best_index_])
        },
        'test_metrics': evaluate_predictions(y_test, test_scores, selected_threshold),
        'cv_summary': summarize_cv(search.cv_results_, search.best_index_)
    }


def choose_best_result(result_list, refit_metric):
    valid = [item for item in result_list if item is not None]
    if not valid:
        raise ValueError('No valid model results were produced')
    return max(valid, key=lambda item: item['validation_metrics'][refit_metric])


def save_model(path_value, trained_model):
    path_value.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_model, path_value)
