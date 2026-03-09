import os
from tqdm import tqdm
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV,LeaveOneGroupOut
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
    def __init__(self, X_all, y_all, groups_all,fs):
        self.X_all = X_all
        self.y_all = y_all
        self.groups_all = groups_all
        self.fs = fs

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

        m0, m1 = 0,0

        if strategy == 'min':
            m = min(N0, N1)
            print(f"策略: 取较小类别全部，m={m}")
        elif strategy == 'ratio':
            m0 = int(N0 * ratio)
            m1 = int(N1 * ratio)
            print(f"策略: 按比例 {ratio:.1%}")
        elif strategy == 'fixed':
            if m is None:
                raise ValueError("strategy='fixed' 时必须指定 m 参数")
            if m > min(N0, N1):
                raise ValueError(f"m={m} 超过较小类别样本数 {min(N0, N1)}")
            print(f"策略: 固定样本数，m={m}")
        else:
            raise ValueError(f"未知策略: {strategy}")

        X_train = np.vstack([self.rest_data[:m0], self.task_data[:m1]])
        y_train = np.concatenate([self.rest_labels[:m0], self.task_labels[:m1]])

        # 剩余作为测试集
        X_test = np.vstack([self.rest_data[m0:], self.task_data[m1:]])
        y_test = np.concatenate([self.rest_labels[m0:], self.task_labels[m1:]])

        print(f"训练集: {X_train.shape[0]} 样本")
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

    def cross_validate_logo(self):
        logo = LeaveOneGroupOut()
        results = {}
        subject_ids = np.unique(self.groups_all)

        print(f"\n共 {len(subject_ids)} 个被试")
        print(f"被试ID: {list(subject_ids)}")

        for i, (train_idx, test_idx) in enumerate(logo.split(self.X_all, self.y_all,self.groups_all)):
            print(f"Fold {i}:")
            print(f"  Train: index={train_idx}, group={groups_all[train_idx]}")
            print(f"  Test:  index={test_idx}, group={groups_all[test_idx]}")

            X_train = self.X_all[train_idx]
            y_train = self.y_all[train_idx]
            X_test = self.X_all[test_idx]
            y_test = self.y_all[test_idx]

            print(f"训练样本: {X_train.shape[0]} | 测试样本: {X_test.shape[0]}")
            res = self.train_eval_one_split(X_train, y_train, X_test, y_test)
            results[i] = res

            print(f"测试集 AUC: {res['test']['auc']:.4f}")
            print(f"准确率: {res['test']['bal_acc']:.4f}")

            # 汇总结果
            aucs = [results[s]['test']['auc'] for s in subject_ids]
            baccs = [results[s]['test']['bal_acc'] for s in subject_ids]
            f1s = [results[s]['test']['f1'] for s in subject_ids]

            print("跨被试最终结果（被试平均）")
            print(f"AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
            print(f"Bal Acc:  {np.mean(baccs):.4f} ± {np.std(baccs):.4f}")
            print(f"F1:       {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

        return results


if __name__ == '__main__':
    # 1. 数据文件夹
    PROCESSED_DATASET_PATH = '../../data/processed/'
    processed_files = [os.path.join(PROCESSED_DATASET_PATH,f) for f in os.listdir(PROCESSED_DATASET_PATH)]

    # 2. 加载所有被试数据，并分配 group
    X_list, y_list, group_list = [], [], []
    fs = None

    for subj_idx, file_path in enumerate(tqdm(processed_files, desc="加载被试数据")):
        data = joblib.load(file_path)
        rest_data = data["datasets"][0]
        task_data = data["datasets"][1]

        # 标签
        rest_label = np.zeros(len(rest_data))
        task_label = np.ones(len(task_data))

        # 拼接
        X_subj = np.vstack([rest_data, task_data])
        y_subj = np.hstack([rest_label, task_label])
        group_subj = np.full(len(X_subj), subj_idx)

        X_list.append(X_subj)
        y_list.append(y_subj)
        group_list.append(group_subj)

        if fs is None:
            fs = data["fs"]

    # 合并成全局数据
    X_all = np.vstack(X_list)
    y_all = np.hstack(y_list)
    groups_all = np.hstack(group_list)

    print(f"\n样本: {X_all.shape[0]}, 被试数: {len(np.unique(groups_all))}")

    # 3. 模型 + 跨被试验证
    fm = FeatureModel(X_all, y_all, groups_all, fs)
    results = fm.cross_validate_logo()

    last_subj = 4
    Visualizer.plot_auc(
        results[last_subj]['test']['fpr'],
        results[last_subj]['test']['tpr'],
        results[last_subj]['test']['auc']
    )
    Visualizer.plot_confusion_matrix(results[last_subj]['test']['cm'])