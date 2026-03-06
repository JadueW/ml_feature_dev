from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score, confusion_matrix, roc_curve
)

from src.visualize.visualizer import Visualizer

import shap
import numpy as np
import warnings
from tqdm import trange  # 导入 tqdm

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

N_CHANNELS = 128
N_BANDS = 6
N_TYPES = 2  # 0=abs, 1=rel
N_FEATURES = N_CHANNELS * N_BANDS * N_TYPES

class FeatureModel:
    def __init__(self, datasets, label_mapping, fs):
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
        :param ratio: float, 默认 0.8, 当 strategy='ratio' 时使用
        :param m: 当 strategy='fixed' 时使用，指定每个类别取多少样本
        :return: X_train, y_train, X_test, y_test
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

    def train_eval_one_split(self, X_train, y_train, X_test, y_test, inner_cv_splits=5):
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
            return_train_score=True
        )
        gs.fit(X_train, y_train)
        best_model = gs.best_estimator_
        best_score = gs.best_score_

        # 训练集上的预测
        y_train_prob = best_model.predict_proba(X_train)[:, 1]
        y_train_pred = (y_train_prob >= 0.5).astype(int)

        # 测试集上的预测
        y_test_prob = best_model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_prob >= 0.5).astype(int)

        # 训练集指标
        train_auc = roc_auc_score(y_train, y_train_prob)
        train_bal_acc = balanced_accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        train_cm = confusion_matrix(y_train, y_train_pred)
        train_fpr, train_tpr, _ = roc_curve(y_train, y_train_prob)

        # 测试集指标
        test_auc = roc_auc_score(y_test, y_test_prob)
        test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_cm = confusion_matrix(y_test, y_test_pred)
        test_fpr, test_tpr, _ = roc_curve(y_test, y_test_prob)

        # 存储训练集结果
        eval_result['train'] = {
            'auc': float(train_auc),
            'bal_acc': float(train_bal_acc),
            'f1': float(train_f1),
            'cm': train_cm,
            'fpr': train_fpr,
            'tpr': train_tpr,
            'best_params': gs.best_params_,
            'cv_best_score': float(best_score)
        }

        # 存储测试集结果
        eval_result['test'] = {
            'auc': float(test_auc),
            'bal_acc': float(test_bal_acc),
            'f1': float(test_f1),
            'cm': test_cm,
            'fpr': test_fpr,
            'tpr': test_tpr
        }

        # 存储 cv_results 用于后续可视化
        eval_result['cv_results'] = gs.cv_results_

        # 为兼容原有代码，保留顶层结果（测试集为主）
        eval_result['auc'] = float(test_auc)
        eval_result['bal_acc'] = float(test_bal_acc)
        eval_result['f1'] = float(test_f1)
        eval_result['cm'] = test_cm
        eval_result['best_params'] = gs.best_params_
        eval_result['best_score'] = float(best_score)
        eval_result['fpr'] = test_fpr
        eval_result['tpr'] = test_tpr
        eval_result['best_model'] = best_model

        return eval_result

    def train_eval_splits(self, X_train, y_train, X_test, y_test, n_splits=5):

        n_splits_eval_results = {}
        for i in trange(n_splits, desc="Training splits"):
            eval_result = self.train_eval_one_split(X_train, y_train, X_test, y_test)
            n_splits_eval_results[i] = eval_result

        best_eval_result = max(n_splits_eval_results.values(), key=lambda x: x['best_score'])
        return best_eval_result, n_splits_eval_results


if __name__ == '__main__':
    data_path = '../../data/processed/test.pkl'
    import joblib

    dataset = joblib.load(data_path)
    datasets = dataset['datasets']
    label_mapping = dataset['label_mapping']
    fs = dataset['fs']

    fm = FeatureModel(datasets, label_mapping, fs)
    X_train, y_train, X_test, y_test = fm.train_test_split_manual(strategy='ratio')

    # 获取最佳结果和所有结果
    best_eval_result, all_eval_results = fm.train_eval_splits(X_train, y_train, X_test, y_test,n_splits=1)

    print("\n最佳评估结果对比:")
    print("-" * 50)
    print("训练集表现:")
    print(f"  AUC: {best_eval_result['train']['auc']:.4f}")
    print(f"  平衡准确率: {best_eval_result['train']['bal_acc']:.4f}")
    print(f"  F1: {best_eval_result['train']['f1']:.4f}")
    print(f"  交叉验证得分: {best_eval_result['train']['cv_best_score']:.4f}")

    print("\n测试集表现:")
    print(f"  AUC: {best_eval_result['test']['auc']:.4f}")
    print(f"  平衡准确率: {best_eval_result['test']['bal_acc']:.4f}")
    print(f"  F1: {best_eval_result['test']['f1']:.4f}")
    print("-" * 50)

    # 检查过拟合情况
    train_auc = best_eval_result['train']['auc']
    test_auc = best_eval_result['test']['auc']
    gap = train_auc - test_auc
    print(f"训练集-测试集 AUC 差距: {gap:.4f}")
    if gap > 0.1:
        print("⚠️ 可能存在过拟合，差距较大")
    elif gap < -0.05:
        print("⚠️ 测试集表现优于训练集，数据分布可能不一致")
    else:
        print("✅ 模型泛化能力良好")

    print(f"\n最佳参数: {best_eval_result['best_params']}")
    print("混淆矩阵:")
    print(best_eval_result['cm'])

    # 绘制最佳split的交叉验证结果
    print("\n绘制最佳split的交叉验证结果...")
    Visualizer.plot_cv_results(best_eval_result)

    # 绘制所有split的对比
    print("\n绘制所有split的交叉验证结果对比...")
    Visualizer.plot_all_splits_cv_results(all_eval_results)

    fpr = best_eval_result['fpr']
    tpr = best_eval_result['tpr']
    auc = best_eval_result['auc']
    cm = best_eval_result['cm']

    Visualizer.plot_auc(fpr, tpr, auc)
    Visualizer.plot_confusion_matrix(cm)