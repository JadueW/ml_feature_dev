import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap

plt.rcParams.update({
    # --- 字体设置 ---
    'font.family': 'sans-serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif', 'SimHei', 'Microsoft YaHei UI'],
    'mathtext.fontset': 'stix',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.unicode_minus': False,

    # --- 字号设置---
    'font.size': 16,
    'axes.titlesize': 13,
    'axes.labelsize': 13,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 10,
    'figure.titlesize': 11,

    # --- 线条和标记 ---
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'lines.markeredgewidth': 0.5,

    # 坐标轴样式
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.0,

    # --- 刻度样式 ---
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': False,
    'ytick.right': False,

    # --- 图例 ---
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'legend.borderpad': 0.4,
    'legend.handlelength': 1.5,

    # --- 输出 ---
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,

    'axes.prop_cycle': plt.cycler(color=[
        '#1A2CA3', '#C40C0C', '#2e7d32', '#f57c00',
        '#6a1b9a', '#00695c', '#ad1457', '#455a64',
    ]),
})

class Visualizer:
    """ 绘图工具类 """
    BAND_NAMES = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']

    @classmethod
    def plot_auc(cls,fpr,tpr,auc):
        """  绘制auc """
        plt.figure(figsize=(15,8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR (False Positive Rate)')
        plt.ylabel('TPR (True Positive Rate)')
        plt.title('AUC_ROC')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()



    @classmethod
    def plot_confusion_matrix(cls,confusion_matrix):
        plt.figure(figsize=(10,10))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='coolwarm', cbar=False,
                    xticklabels=['Neg.', 'Pos.'], yticklabels=['Neg.', 'Pos.'])
        plt.xlabel('Predict Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')

        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_cv_results(cls, eval_result):
        """ 绘制单次交叉验证结果 """
        cv_results = eval_result['cv_results']

        # 获取参数组合的索引
        n_params = len(cv_results['params'])
        param_combinations = range(n_params)

        # 获取训练和测试的均值和标准差
        mean_train_score = cv_results['mean_train_score']
        std_train_score = cv_results['std_train_score']
        mean_test_score = cv_results['mean_test_score']
        std_test_score = cv_results['std_test_score']

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # 第一个子图：训练集和测试集得分对比（带误差线）
        x = np.arange(n_params)
        width = 0.35

        ax1.errorbar(x - width / 2, mean_train_score, yerr=std_train_score,
                     fmt='o-', capsize=5, capthick=2, elinewidth=2, markersize=8,
                     label='Trainset Scores(KFold)', color='blue', alpha=0.7)
        ax1.errorbar(x + width / 2, mean_test_score, yerr=std_test_score,
                     fmt='s-', capsize=5, capthick=2, elinewidth=2, markersize=8,
                     label='Testset Scores(KFold)', color='red', alpha=0.7)

        # 添加测试集最终得分（水平线）
        test_auc = eval_result['test']['auc']
        ax1.axhline(y=test_auc, color='green', linestyle='--', linewidth=2,
                    label=f'Trainset Final AUC: {test_auc:.4f}')

        ax1.set_xlabel('Parameter Combined Index')
        ax1.set_ylabel('AUC Scores')
        ax1.set_title('KFold Parameter Combined Performance(With Error Line)')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Combine {i + 1}' for i in x], rotation=45)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.5, 1.05])

        # 第二个子图：所有fold的详细表现
        # 获取fold数量
        fold_keys = [k for k in cv_results.keys() if k.startswith('split') and k.endswith('_test_score')]
        n_splits = len(fold_keys)

        # 为每个参数组合绘制所有fold的得分
        for param_idx in range(n_params):
            train_fold_scores = []
            test_fold_scores = []

            # 收集该参数组合在所有fold的得分
            for fold in range(n_splits):
                train_key = f'split{fold}_train_score'
                test_key = f'split{fold}_test_score'

                if train_key in cv_results and test_key in cv_results:
                    train_fold_scores.append(cv_results[train_key][param_idx])
                    test_fold_scores.append(cv_results[test_key][param_idx])

            if train_fold_scores:  # 如果有数据
                # 添加抖动以避免重叠
                x_train = np.ones(len(train_fold_scores)) * (param_idx - 0.2) + np.random.normal(0, 0.03,
                                                                                                 len(train_fold_scores))
                x_test = np.ones(len(test_fold_scores)) * (param_idx + 0.2) + np.random.normal(0, 0.03,
                                                                                               len(test_fold_scores))

                ax2.scatter(x_train, train_fold_scores, alpha=0.5, color='lightblue', s=30, marker='.',
                            label='训练集各fold' if param_idx == 0 else "")
                ax2.scatter(x_test, test_fold_scores, alpha=0.5, color='lightcoral', s=30, marker='.',
                            label='验证集各fold' if param_idx == 0 else "")

        # 添加均值线和误差线
        ax2.errorbar(x - width / 2, mean_train_score, yerr=std_train_score,
                     fmt='o', capsize=3, color='blue', elinewidth=2, markersize=6, label='训练集均值')
        ax2.errorbar(x + width / 2, mean_test_score, yerr=std_test_score,
                     fmt='s', capsize=3, color='red', elinewidth=2, markersize=6, label='验证集均值')

        ax2.set_xlabel('参数组合索引')
        ax2.set_ylabel('AUC 得分')
        ax2.set_title('各参数组合在不同fold的详细表现（点状分布）')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'组合 {i + 1}' for i in x], rotation=45)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0.5, 1.05])

        plt.tight_layout()
        plt.show()

        # 打印最佳参数组合的详细信息
        best_idx = cv_results['rank_test_score'].argmin()
        print(f"\n最佳参数组合 (索引 {best_idx + 1}):")
        print(f"参数: {cv_results['params'][best_idx]}")
        print(f"训练集平均 AUC: {mean_train_score[best_idx]:.4f} ± {std_train_score[best_idx]:.4f}")
        print(f"验证集平均 AUC: {mean_test_score[best_idx]:.4f} ± {std_test_score[best_idx]:.4f}")
        print(f"测试集最终 AUC: {eval_result['test']['auc']:.4f}")

    @classmethod
    def plot_all_splits_cv_results(cls,n_splits_eval_results):
        n_splits = len(n_splits_eval_results)

        # 根据结果数量动态调整子图布局
        n_cols = min(3, n_splits)  # 最多3列
        n_rows = (n_splits + n_cols - 1) // n_cols  # 计算需要的行数

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])  # 确保 axes 始终是可索引的
        axes = axes.flatten()

        # 为每个 split 绘制子图
        for i, (split_idx, eval_result) in enumerate(n_splits_eval_results.items()):
            if i >= len(axes):  # 防止超出子图数量
                break

            cv_results = eval_result['cv_results']

            # 获取数据
            mean_train_score = cv_results['mean_train_score']
            std_train_score = cv_results['std_train_score']
            mean_test_score = cv_results['mean_test_score']
            std_test_score = cv_results['std_test_score']

            x = np.arange(len(mean_train_score))
            width = 0.35

            # 绘制带误差线的柱状图
            axes[i].bar(x - width / 2, mean_train_score, width,
                        yerr=std_train_score, capsize=5,
                        label='Trainset (CV)', color='skyblue', alpha=0.7,
                        error_kw={'elinewidth': 2, 'ecolor': 'darkblue'})

            axes[i].bar(x + width / 2, mean_test_score, width,
                        yerr=std_test_score, capsize=5,
                        label='Testset (CV)', color='lightcoral', alpha=0.7,
                        error_kw={'elinewidth': 2, 'ecolor': 'darkred'})

            # 添加测试集最终得分（水平线）
            test_auc = eval_result['test']['auc']
            axes[i].axhline(y=test_auc, color='green', linestyle='--', linewidth=2,
                            label=f'Testset: {test_auc:.3f}')

            # 标记最佳参数组合
            best_idx = cv_results['rank_test_score'].argmin()
            axes[i].plot(best_idx, mean_test_score[best_idx] + std_test_score[best_idx] + 0.02,
                         'r*', markersize=15, label='Best Combine')

            # 设置标题和标签
            axes[i].set_title(f'Train Iter {split_idx + 1}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Parameter Combine', fontsize=10)
            axes[i].set_ylabel('AUC Scores', fontsize=10)
            axes[i].set_xticks(x)
            axes[i].set_xticklabels([f'P{i + 1}' for i in x], rotation=45, fontsize=8)
            axes[i].set_ylim([0.4, 1.05])
            axes[i].grid(True, alpha=0.3, axis='y')
            axes[i].legend(loc='lower right', fontsize=8, framealpha=0.9)

            # 在柱子上方标注具体数值
            for j in x:
                axes[i].text(j - width / 2, mean_train_score[j] + std_train_score[j] + 0.01,
                             f'{mean_train_score[j]:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)
                axes[i].text(j + width / 2, mean_test_score[j] + std_test_score[j] + 0.01,
                             f'{mean_test_score[j]:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

        # 隐藏多余的子图
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle('Repeat Train KFold Results Compare', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

        # 打印统计摘要
        print("\n" + "=" * 60)
        print("多次训练结果统计摘要")
        print("=" * 60)

        all_test_aucs = [result['test']['auc'] for result in n_splits_eval_results.values()]
        all_train_aucs = [result['train']['auc'] for result in n_splits_eval_results.values()]
        all_cv_scores = [result['train']['cv_best_score'] for result in n_splits_eval_results.values()]

        print(f"测试集 AUC: 均值={np.mean(all_test_aucs):.4f} ± {np.std(all_test_aucs):.4f}")
        print(f"训练集 AUC: 均值={np.mean(all_train_aucs):.4f} ± {np.std(all_train_aucs):.4f}")
        print(f"交叉验证得分: 均值={np.mean(all_cv_scores):.4f} ± {np.std(all_cv_scores):.4f}")

        # 稳定性分析
        cv_std = np.std(all_test_aucs)
        if cv_std < 0.01:
            stability = "非常稳定"
        elif cv_std < 0.02:
            stability = "较稳定"
        elif cv_std < 0.03:
            stability = "一般稳定"
        else:
            stability = "不稳定"

        print(f"\n模型稳定性: {stability} (标准差={cv_std:.4f})")

        # 找出最佳和最差结果
        best_idx = np.argmax(all_test_aucs)
        worst_idx = np.argmin(all_test_aucs)
        print(f"\n最佳结果: 轮次 {best_idx + 1}, AUC={all_test_aucs[best_idx]:.4f}")
        print(f"最差结果: 轮次 {worst_idx + 1}, AUC={all_test_aucs[worst_idx]:.4f}")

    # 下面为基于SHAP值的可视化
    @classmethod
    def plot_shap_channel_importance(cls, shap_values, n_channels=128, n_bands=6, n_types=2,
                                     data_name='test', figsize=(12, 8)):
        """
        绘制通道重要性：每个通道的平均绝对 SHAP 值（跨频段和类型）

        参数:
        - shap_values: SHAP 值数组，形状可以是 (n_samples, n_features) 或 (n_samples, n_channels, n_bands, n_types)
        - n_channels, n_bands, n_types: 数据维度
        - data_name: 数据集名称（用于标题）
        """
        # 确保 shap_values 是 (n_samples, n_channels, n_bands, n_types)
        if shap_values.ndim == 2:
            shap_reshaped = shap_values.reshape(-1, n_channels, n_bands * n_types).reshape(-1, n_channels, n_bands,
                                                                                           n_types)
        else:
            shap_reshaped = shap_values

        # 计算每个通道的平均绝对 SHAP 值
        channel_imp = np.mean(np.abs(shap_reshaped), axis=(0, 2, 3))
        sorted_idx = np.argsort(channel_imp)[::-1]
        sorted_imp = channel_imp[sorted_idx]

        plt.figure(figsize=figsize)
        plt.bar(range(n_channels), sorted_imp, color='steelblue')
        plt.xlabel('通道 (按重要性排序)')
        plt.ylabel('平均 |SHAP| 值')
        plt.title(f'通道重要性（{data_name} 集）')
        plt.grid(axis='y', alpha=0.3)

        # 标注前10个通道的原始索引
        for i in range(min(10, n_channels)):
            plt.text(i, sorted_imp[i] + 0.001 * max(sorted_imp), f'{sorted_idx[i]}',
                     ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.show()
        return sorted_idx, sorted_imp

    @classmethod
    def plot_shap_band_importance(cls, shap_values, n_channels=128, n_bands=6, n_types=2,
                                  separate_types=True, data_name='test', figsize=(10, 6),band_names = None):
        """
        绘制频段重要性：每个频段的平均绝对 SHAP 值
        separate_types: 是否区分 abs/rel
        """
        if band_names is None:
            band_names = cls.BAND_NAMES

        if shap_values.ndim == 2:
            shap_reshaped = shap_values.reshape(-1, n_channels, n_bands * n_types).reshape(-1, n_channels, n_bands,
                                                                                           n_types)
        else:
            shap_reshaped = shap_values

        if separate_types:
            imp_abs = np.mean(np.abs(shap_reshaped[:, :, :, 0]), axis=(0, 1))
            imp_rel = np.mean(np.abs(shap_reshaped[:, :, :, 1]), axis=(0, 1))

            x = np.arange(n_bands)
            width = 0.35
            plt.figure(figsize=figsize)
            plt.bar(x - width / 2, imp_abs, width, label='abs', color='skyblue')
            plt.bar(x + width / 2, imp_rel, width, label='rel', color='lightcoral')
            plt.xlabel('频段')
            plt.ylabel('平均 |SHAP| 值')
            plt.title(f'频段重要性（{data_name} 集）')
            plt.xticks(x, band_names)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()
            return imp_abs, imp_rel
        else:
            imp = np.mean(np.abs(shap_reshaped), axis=(0, 1, 3))
            plt.figure(figsize=figsize)
            plt.bar(range(n_bands), imp, color='steelblue')
            plt.xlabel('频段')
            plt.ylabel('平均 |SHAP| 值')
            plt.title(f'频段重要性（{data_name} 集）')
            plt.xticks(range(n_bands), band_names)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()
            return imp

    @classmethod
    def plot_shap_type_importance(cls, shap_values, n_channels=128, n_bands=6, n_types=2,
                                  data_name='test', figsize=(6, 5)):
        """ 绘制 abs 与 rel 类型的重要性对比 """
        if shap_values.ndim == 2:
            shap_reshaped = shap_values.reshape(-1, n_channels, n_bands * n_types).reshape(-1, n_channels, n_bands,
                                                                                           n_types)
        else:
            shap_reshaped = shap_values

        imp_abs = np.mean(np.abs(shap_reshaped[:, :, :, 0]))
        imp_rel = np.mean(np.abs(shap_reshaped[:, :, :, 1]))

        plt.figure(figsize=figsize)
        bars = plt.bar(['abs', 'rel'], [imp_abs, imp_rel], color=['skyblue', 'lightcoral'])
        plt.ylabel('平均 |SHAP| 值')
        plt.title(f'类型重要性对比（{data_name} 集）')

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.001 * max(imp_abs, imp_rel),
                     f'{height:.4f}', ha='center', va='bottom')

        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        return imp_abs, imp_rel

    @classmethod
    def plot_shap_channel_band_heatmap(cls, shap_values, n_channels=128, n_bands=6, n_types=2,
                                       type_idx=None, data_name='test', figsize=(14, 10),band_names=None):
        """
        绘制通道-频段热图：每个通道每个频段的平均绝对 SHAP 值
        type_idx: None 合并类型；0 仅 abs；1 仅 rel
        """
        if band_names is None:
            band_names = cls.BAND_NAMES

        if shap_values.ndim == 2:
            shap_reshaped = shap_values.reshape(-1, n_channels, n_bands * n_types).reshape(-1, n_channels, n_bands,
                                                                                           n_types)
        else:
            shap_reshaped = shap_values

        if type_idx is None:
            heatmap_data = np.mean(np.abs(shap_reshaped), axis=(0, 3))  # (ch, band)
            title_suffix = '合并类型'
        else:
            type_name = 'abs' if type_idx == 0 else 'rel'
            heatmap_data = np.mean(np.abs(shap_reshaped[:, :, :, type_idx]), axis=0)
            title_suffix = type_name

        plt.figure(figsize=figsize)
        sns.heatmap(heatmap_data.T, cmap='viridis', cbar_kws={'label': '平均 |SHAP|'})
        plt.xlabel('通道')
        plt.ylabel('频段')
        plt.title(f'通道-频段重要性热图 ({title_suffix}, {data_name} 集)')
        plt.yticks(ticks=np.arange(n_bands)+0.5, labels=band_names)
        plt.tight_layout()
        plt.show()
        return heatmap_data

    @classmethod
    def plot_shap_direction_summary(cls, shap_values, n_channels=128, n_bands=6, n_types=2,
                                    top_k_channels=10, data_name='test', figsize=(15, 10)):
        """
        绘制方向性总结：top 通道的 SHAP 值分布（箱线图），展示正负贡献
        """
        if shap_values.ndim == 2:
            shap_reshaped = shap_values.reshape(-1, n_channels, n_bands * n_types).reshape(-1, n_channels, n_bands,
                                                                                           n_types)
        else:
            shap_reshaped = shap_values

        # 每个通道的平均 SHAP（不是绝对值）
        channel_mean_shap = np.mean(shap_reshaped, axis=(0, 2, 3))
        top_channels = np.argsort(np.abs(channel_mean_shap))[-top_k_channels:][::-1]

        data_for_box = []
        channel_labels = []
        for ch in top_channels:
            ch_shap = shap_reshaped[:, ch, :, :].reshape(-1)  # 展平所有样本、频段、类型
            data_for_box.append(ch_shap)
            channel_labels.append(f'Ch{ch}\n(mean={channel_mean_shap[ch]:.3f})')

        plt.figure(figsize=figsize)
        bp = plt.boxplot(data_for_box, labels=channel_labels, patch_artist=True,
                         showfliers=False, vert=True)

        # 着色：均值正为绿色，负为红色
        for i, ch in enumerate(top_channels):
            if channel_mean_shap[ch] > 0:
                bp['boxes'][i].set_facecolor('lightgreen')
            else:
                bp['boxes'][i].set_facecolor('lightcoral')

        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.xlabel('通道')
        plt.ylabel('SHAP 值分布')
        plt.title(f'Top {top_k_channels} 通道的 SHAP 值分布（{data_name} 集）\n绿色正贡献主导，红色负贡献主导')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 打印方向解释
        print("\n方向性解释：")
        for ch in top_channels:
            mean_val = channel_mean_shap[ch]
            direction = "正类" if mean_val > 0 else "负类"
            print(f"通道 {ch}: 平均 SHAP = {mean_val:.4f}，对预测为 {direction} 有贡献")

        return top_channels, channel_mean_shap[top_channels]

    @classmethod
    def plot_shap_channel_profile(cls, shap_values, channel_idx, n_channels=128, n_bands=6, n_types=2,
                                  data_name='test', figsize=(12, 5),band_names=None):
        """
        绘制指定通道内频段和类型的平均 SHAP 值及其分布
        """
        if band_names is None:
            band_names=cls.BAND_NAMES

        if shap_values.ndim == 2:
            shap_reshaped = shap_values.reshape(-1, n_channels, n_bands * n_types).reshape(-1, n_channels, n_bands,
                                                                                           n_types)
        else:
            shap_reshaped = shap_values

        ch_shap = shap_reshaped[:, channel_idx, :, :]  # (样本, 频段, 类型)
        avg_shap = np.mean(ch_shap, axis=0)  # (频段, 类型)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 条形图：abs 和 rel 分开
        bands = np.arange(n_bands)
        width = 0.35
        ax1.bar(bands - width / 2, avg_shap[:, 0], width, label='abs', color='skyblue')
        ax1.bar(bands + width / 2, avg_shap[:, 1], width, label='rel', color='lightcoral')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('频段')
        ax1.set_ylabel('平均 SHAP 值')
        ax1.set_title(f'通道 {channel_idx} 平均 SHAP')
        ax1.set_xticks(bands)
        ax1.set_xticklabels(band_names)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # 箱线图：所有样本、所有频段类型的 SHAP 分布
        data_flat = ch_shap.reshape(-1)
        ax2.boxplot(data_flat, vert=False, showfliers=False)
        ax2.axvline(x=0, color='black', linestyle='--')
        ax2.set_xlabel('SHAP 值')
        ax2.set_title(f'通道 {channel_idx} SHAP 分布')
        ax2.set_yticks([])

        plt.suptitle(f'通道 {channel_idx} SHAP 分析（{data_name} 集）')
        plt.tight_layout()
        plt.show()
        return avg_shap

    @classmethod
    def plot_shap_summary(cls, shap_values, X, feature_names=None, max_display=20, data_name='test'):
        """
        使用 SHAP 内置的 summary_plot
        """
        # 确保 shap_values 是二维 (样本, 特征)
        if hasattr(shap_values, 'values'):  # 如果是 Explanation 对象
            shap_values = shap_values.values
        if shap_values.ndim == 4:  # (样本, 通道, 频段, 类型)
            shap_values = shap_values.reshape(shap_values.shape[0], -1)
        elif shap_values.ndim == 3 and shap_values.shape[2] == 2:  # 可能是 (样本, 特征, 类别)
            shap_values = shap_values[:, :, 1]  # 取正类

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names, max_display=max_display, show=False)
        plt.title(f'SHAP Summary Plot ({data_name} 集)')
        plt.tight_layout()
        plt.show()

