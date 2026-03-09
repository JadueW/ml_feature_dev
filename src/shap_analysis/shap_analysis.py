import shap
import numpy as np
from src.visualize.visualizer import Visualizer

try:
    from shap import KernelExplainer, Explainer, LinearExplainer
    _has_shap_classes = True
except ImportError:
    _has_shap_classes = False
    print("警告: 无法直接导入 SHAP 类，将使用 shap.xxx 的方式")

class ShapAnalyzer:

    def __init__(self, model, X_train, X_test, n_channels=128, n_bands=6, n_types=2):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.n_types = n_types
        self.n_features = n_channels * n_bands * n_types

        # 验证特征数
        assert X_train.shape[1] == self.n_features, \
            f"特征数不匹配：训练集特征数 {X_train.shape[1]}，期望 {self.n_features}"
        assert X_test.shape[1] == self.n_features, \
            f"特征数不匹配：测试集特征数 {X_test.shape[1]}，期望 {self.n_features}"

        # 生成特征标签（用于 summary_plot）
        self.feature_names = []
        for ch in range(n_channels):
            for b in range(n_bands):
                self.feature_names.append(f'ch{ch}_b{b}_abs')
            for b in range(n_bands):
                self.feature_names.append(f'ch{ch}_b{b}_rel')

        self.shap_values_train = None
        self.shap_values_test = None
        self.explainer = None

        # 生成特定的特证名 channel_type_band 用于后续排序
        self.band_names = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']
        self.feature_names = []
        for ch in range(n_channels):
            for b in range(n_bands):
                self.feature_names.append(f'ch{ch}_abs_{self.band_names[b]}')
            for b in range(n_bands):
                self.feature_names.append(f'ch{ch}_rel_{self.band_names[b]}')

    def compute_shap(self, background_size=100, use_kernel=False):
        """ 计算训练集和测试集的 SHAP 值 """
        # 选取背景样本
        if len(self.X_train) > background_size:
            idx = np.random.choice(len(self.X_train), background_size, replace=False)
            background = self.X_train[idx]
        else:
            background = self.X_train

        print("创建 SHAP 解释器...")

        # 对于 Pipeline 模型，直接使用 predict_proba 作为可调用函数
        if hasattr(self.model, 'named_steps') and hasattr(self.model, 'predict_proba'):
            # 这是一个 sklearn Pipeline
            if use_kernel:
                print("使用 KernelExplainer（较慢但准确）")
                if _has_shap_classes:
                    self.explainer = KernelExplainer(self.model.predict_proba, background)
                else:
                    self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
            else:
                print("使用通用 Explainer (基于 predict_proba)")
                # 使用 predict_proba 作为模型函数
                if _has_shap_classes:
                    self.explainer = Explainer(self.model.predict_proba, background)
                else:
                    self.explainer = shap.Explainer(self.model.predict_proba, background)
        else:
            # 非Pipeline模型，尝试使用特定解释器
            if use_kernel:
                if _has_shap_classes:
                    self.explainer = KernelExplainer(self.model.predict_proba, background)
                else:
                    self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
            else:
                # 尝试使用 LinearExplainer（针对线性模型）
                try:
                    if _has_shap_classes:
                        self.explainer = LinearExplainer(self.model, background)
                    else:
                        self.explainer = shap.LinearExplainer(self.model, background)
                    print("使用 LinearExplainer（针对线性模型优化）")
                except Exception as e:
                    print(f"LinearExplainer 不可用 ({e})，使用通用 Explainer")
                    if _has_shap_classes:
                        self.explainer = Explainer(self.model.predict_proba, background)
                    else:
                        self.explainer = shap.Explainer(self.model.predict_proba, background)

        print("计算训练集 SHAP 值...")
        try:
            self.shap_values_train = self.explainer.shap_values(self.X_train)
        except:
            self.shap_values_train = self.explainer(self.X_train)

        print("计算测试集 SHAP 值...")
        try:
            self.shap_values_test = self.explainer.shap_values(self.X_test)
        except:
            self.shap_values_test = self.explainer(self.X_test)

        return self

    def _extract_shap_array(self, shap_values):
        """ 从 Explanation 对象或数组中提取正类的二维 SHAP 数组 """
        if hasattr(shap_values, 'values'):
            arr = shap_values.values
        else:
            arr = shap_values

        if arr.ndim == 3:  # (样本, 特征, 类别)
            arr = arr[:, :, 1]  # 取正类
        return arr

    def get_shap_arrays(self, data='test'):
        """
        返回 (shap_array, X_array) 均为 (样本, 特征) 的原始形状
        """
        if data == 'train':
            shap_vals = self.shap_values_train
            X_vals = self.X_train
        else:
            shap_vals = self.shap_values_test
            X_vals = self.X_test

        shap_arr = self._extract_shap_array(shap_vals)
        return shap_arr, X_vals

    def get_shap_reshaped(self, data='test'):
        """
        返回 reshape 后的 (样本, 通道, 频段, 类型) 的 SHAP 值和特征值
        """
        shap_arr, X_arr = self.get_shap_arrays(data)
        shap_reshaped = shap_arr.reshape(-1, self.n_channels, self.n_bands * self.n_types)
        shap_reshaped = shap_reshaped.reshape(-1, self.n_channels, self.n_bands, self.n_types)
        X_reshaped = X_arr.reshape(-1, self.n_channels, self.n_bands * self.n_types)
        X_reshaped = X_reshaped.reshape(-1, self.n_channels, self.n_bands, self.n_types)
        return shap_reshaped, X_reshaped


    def plot_channel_importance(self, data='test', **kwargs):
        shap_arr, _ = self.get_shap_arrays(data)
        return Visualizer.plot_shap_channel_importance(
            shap_arr, self.n_channels, self.n_bands, self.n_types,
            data_name=data, **kwargs
        )

    def plot_band_importance(self, data='test', separate_types=True, **kwargs):
        shap_arr, _ = self.get_shap_arrays(data)
        return Visualizer.plot_shap_band_importance(
            shap_arr, self.n_channels, self.n_bands, self.n_types,
            separate_types=separate_types, data_name=data, **kwargs
        )

    def plot_type_importance(self, data='test', **kwargs):
        shap_arr, _ = self.get_shap_arrays(data)
        return Visualizer.plot_shap_type_importance(
            shap_arr, self.n_channels, self.n_bands, self.n_types,
            data_name=data, **kwargs
        )

    def plot_channel_band_heatmap(self, data='test', type_idx=None, **kwargs):
        shap_arr, _ = self.get_shap_arrays(data)
        return Visualizer.plot_shap_channel_band_heatmap(
            shap_arr, self.n_channels, self.n_bands, self.n_types,
            type_idx=type_idx, data_name=data, **kwargs
        )

    def plot_direction_summary(self, data='test', top_k_channels=10, **kwargs):
        shap_arr, _ = self.get_shap_arrays(data)
        return Visualizer.plot_shap_direction_summary(
            shap_arr, self.n_channels, self.n_bands, self.n_types,
            top_k_channels=top_k_channels, data_name=data, **kwargs
        )

    def plot_channel_profile(self, channel_idx, data='test', **kwargs):
        shap_arr, _ = self.get_shap_arrays(data)
        return Visualizer.plot_shap_channel_profile(
            shap_arr, channel_idx, self.n_channels, self.n_bands, self.n_types,
            data_name=data, **kwargs
        )

    def plot_summary(self, data='test', max_display=20, **kwargs):
        shap_arr, X_arr = self.get_shap_arrays(data)
        return Visualizer.plot_shap_summary(
            shap_arr, X_arr, feature_names=self.feature_names,
            max_display=max_display, data_name=data, **kwargs
        )
