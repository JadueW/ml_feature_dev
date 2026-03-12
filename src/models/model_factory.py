from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

"""
根据配置构建模型和参数网格
"""

def _optional_lightgbm():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier
    except Exception:
        return None


def create_model_pipeline(model_name, model_config):
    class_weight = model_config['class_weight']
    random_state = model_config['random_state']
    steps = []

    if model_config.get('scaler', 'standard') == 'standard' and model_name != 'lightgbm':
        steps.append(('scaler', StandardScaler()))

    if model_name == 'logistic_regression':
        cfg = model_config['logistic_regression']
        estimator = LogisticRegression(
            penalty=cfg['penalty'],
            solver=cfg['solver'],
            max_iter=cfg['max_iter'],
            class_weight=class_weight,
            random_state=random_state
        )
        steps.append(('clf', estimator))
        return Pipeline(steps)

    if model_name == 'linear_svc':
        cfg = model_config['linear_svc']
        estimator = SVC(
            kernel='linear',
            probability=True,
            class_weight=class_weight,
            C=1.0,
            random_state=random_state,
            max_iter=cfg['max_iter']
        )
        steps.append(('clf', estimator))
        return Pipeline(steps)

    if model_name == 'lightgbm':
        lgbm_cls = _optional_lightgbm()
        if lgbm_cls is None:
            return None
        estimator = lgbm_cls(
            objective='binary',
            class_weight=class_weight,
            random_state=random_state,
            verbosity=-1
        )
        steps.append(('clf', estimator))
        return Pipeline(steps)

    raise ValueError('Unsupported model_name: %s' % model_name)


def create_param_grid(model_name, model_config):
    if model_name == 'logistic_regression':
        cfg = model_config['logistic_regression']
        return {
            'clf__C': list(cfg['c_values']),
            'clf__l1_ratio': list(cfg['l1_ratios'])
        }

    if model_name == 'linear_svc':
        cfg = model_config['linear_svc']
        return {
            'clf__C': list(cfg['c_values'])
        }

    if model_name == 'lightgbm':
        cfg = model_config['lightgbm']
        return {
            'clf__num_leaves': list(cfg['num_leaves']),
            'clf__learning_rate': list(cfg['learning_rate']),
            'clf__n_estimators': list(cfg['n_estimators'])
        }

    raise ValueError('Unsupported model_name: %s' % model_name)
