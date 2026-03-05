from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score, confusion_matrix
)

import shap
import numpy as np
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42

N_CHANNELS = 128
N_BANDS = 6
N_TYPES = 2  # 0=abs, 1=rel
N_FEATURES = N_CHANNELS * N_BANDS * N_TYPES

class FeatureModel:
    def __init__(self,datasets,label_mapping,fs):
        self.datasets = datasets
        self.label_mapping = label_mapping
        self.fs = fs

        self.rest_data = self.datasets[0]
        self.task_data = self.datasets[1]
        self.rest_labels = np.ones(self.rest_data.shape[0]) * 0
        self.task_labels = np.ones(self.task_data.shape[0]) * 1


    def train_test_split_manual(self, strategy='min', ratio=0.8, m=None):
        """
        手动划分训练集和测试集
        :param strategy:
            - 'min': 取较小类别的全部样本作为训练集，确保均衡
            - 'ratio': 从每个类别取 ratio 比例的样本（不超过较小类别的数量）
            - 'fixed': 使用指定的 m 值
        :param ratio:
            float, 默认 0.8,当 strategy='ratio' 时使用
        :param m:当 strategy='fixed' 时使用，指定每个类别取多少样本
        :return:X_train, y_train, X_test, y_test
        """
        N0 = self.rest_data.shape[0]  # non_task 样本数
        N1 = self.task_data.shape[0]  # task 样本数

        print(f"样本数量: non_task={N0}, task={N1}")

        if strategy == 'min':
            m = min(N0, N1)
            print(f"策略: 取较小类别全部，m={m}")
        elif strategy == 'ratio':
            m = int(min(N0, N1) * ratio)
            print(f"策略: 按比例 {ratio:.1%}，m={m}")
        elif strategy == 'fixed':
            if m is None:
                raise ValueError("strategy='fixed' 时必须指定 m 参数")
            if m > min(N0, N1):
                raise ValueError(f"m={m} 超过较小类别样本数 {min(N0, N1)}")
            print(f"策略: 固定样本数，m={m}")
        else:
            raise ValueError(f"未知策略: {strategy}")

        X_train = np.vstack([self.rest_data[:m], self.task_data[:m]])
        y_train = np.concatenate([self.rest_labels[:m], self.task_labels[:m]])

        # 剩余作为测试集
        X_test = np.vstack([self.rest_data[m:], self.task_data[m:]])
        y_test = np.concatenate([self.rest_labels[m:], self.task_labels[m:]])

        print(f"训练集: {X_train.shape[0]} 样本 (均衡: {m} non_task + {m} task)")
        print(f"测试集: {X_test.shape[0]} 样本")

        return X_train, y_train, X_test, y_test

    def make_pipeline(self):
        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=20000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])
        return pipe

    def make_param_grid(self):
        return {
            "clf__C": [1e-3, 1e-2, 1e-1, 1, 10, 100],
            "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }

    def train_eval_one_split(self,X_train, y_train, X_test, y_test, inner_cv_splits=5):
        pipe = self.make_pipeline()
        param_grid = self.make_param_grid()
        eval_result = {}

        inner_cv = StratifiedKFold(
            n_splits=inner_cv_splits,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=inner_cv,
            refit=True,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        best_model = gs.best_estimator_

        y_prob = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_prob)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        eval_result['auc'] = float(auc)
        eval_result['bal_acc'] = float(bal_acc)
        eval_result['f1'] = float(f1)
        eval_result['cm'] = cm
        eval_result['best_params'] = gs.best_params_

        return eval_result



if __name__ == '__main__':
    data_path = '../../data/processed/test.pkl'
    import joblib
    dataset = joblib.load(data_path)
    datasets = dataset['datasets']
    label_mapping = dataset['label_mapping']
    fs = dataset['fs']

    fm = FeatureModel(datasets,label_mapping,fs)
    X_train, y_train, X_test, y_test = fm.train_test_split_manual(strategy='ratio')
    eval_result = fm.train_eval_one_split(X_train, y_train, X_test, y_test)

    print(eval_result)
