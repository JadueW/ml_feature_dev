# 项目代码理解与架构梳理

## 1. 项目目标（我对代码的理解）

该项目是一个面向 **uECoG（二分类：non_task vs task）** 的机器学习流水线，目标是：

1. 将原始多段实验数据重组为统一格式；
2. 对时序信号进行滤波与重参考预处理；
3. 提取每个样本的频带功率特征（绝对 + 相对）；
4. 在跨被试场景下训练与验证模型（Leave-One-Group-Out + 内层 GridSearchCV）；
5. 输出可视化结果并保存最佳模型；
6. 预留 SHAP 可解释性分析接口。

---

## 2. 目录与职责映射

- `src/pipeline.py`：主流程编排（数据重构、预处理、特征提取、跨被试训练、可视化、保存模型）。
- `src/preprocess/preprocessor.py`：信号预处理（陷波、带通、CAR）。
- `src/featureExtract/feature_extract.py`：Welch PSD 特征提取（6 频段 × 128 通道 × abs/rel）。
- `src/models/ml_feature_model.py`：模型训练评估与跨被试验证（LogisticRegression + GridSearchCV + LOGO）。
- `src/shap_analysis/shap_analysis.py`：SHAP 解释器封装与数据重排。
- `src/visualize/visualizer.py`：ROC/混淆矩阵/CV/SHAP 等可视化。
- `src/utils/data_check.py`：输入数据完整性检查。

---

## 3. 端到端数据流

### 3.1 原始数据重组（`reconstruct_datasets`）

- 输入：`raw_data['datasets']`（其中第 0 段作为 rest，其余段合并为 task）。
- 操作：
  - 仅保留前 128 通道；
  - 生成标签（rest=0, task=1）；
  - 输出结构为 `{"datasets": {0:(X0,y0),1:(X1,y1)}, "label_mapping", "fs"}`。

### 3.2 预处理（`Preprocessor.preprocess`）

处理顺序：

1. 构建多个陷波滤波器（如 50/100/150Hz）并级联；
2. 构建 1-200Hz 带通滤波器；
3. `sosfiltfilt` 在时间轴做零相位滤波；
4. 全脑平均重参考（CAR）。

输入形状是 `(n_samples, n_channels, n_timepoints)`，输出保持同形状。

### 3.3 特征提取（`FeatureExtractor.compute_psd_features`）

- 用 Welch 估计每个样本、每个通道的 PSD。
- 频段默认：delta/theta/alpha/beta/low_gamma/high_gamma（1-150Hz 内）。
- 绝对功率：各频段平均 PSD。
- 相对功率：`band_power / total_power_linear`（其中 total_power 先 dB 再反变换）。
- 最终特征展平并拼接为：
  - `abs_flat` + `rel_flat`
  - 维度 = `128 * 6 * 2 = 1536`。

### 3.4 跨被试建模（`FeatureModel.cross_validate_logo`）

- 全体被试数据先堆叠成 `X_all, y_all, groups_all`。
- 外层：`LeaveOneGroupOut` 按被试留一；
- 内层：`StratifiedKFold + GridSearchCV` 调参；
- 模型：`StandardScaler + LogisticRegression(solver='saga', class_weight='balanced')`；
- 网格参数：`C × l1_ratio`。

每个外层 fold 会输出 train/test 指标（AUC、bal_acc、F1、CM、ROC 曲线）和最佳模型；最终汇总跨被试均值与方差。

### 3.5 模型保存

- 从所有外层 fold 中选择 `best_score` 最高者的 `best_model`；
- 保存到 `models/cross_subjects_model_YYYYMMDD.pkl`。

---

## 4. 关键实现细节

### 4.1 形状与维度约束

- 预处理阶段：三维时序输入。
- 特征阶段：二维特征输入（每样本 1536 特征）。
- `DataChecker` 明确固定参数：128 通道、6 频段、2 类型。

### 4.2 并行策略

- 特征提取可在样本维度并行（`joblib.Parallel`）；
- 外层跨被试 fold 也支持并行（`cross_validate_logo(n_jobs=-1)`）。

### 4.3 SHAP 兼容逻辑

`ShapAnalyzer` 根据模型类型与可用 API 自动选择 `Explainer/KernelExplainer/LinearExplainer`，并对返回格式（`Explanation` vs ndarray）做统一抽取，最终可 reshape 到 `(samples, channels, bands, types)` 便于神经信号解释。

---

## 5. 我观察到的潜在问题/改进点

1. **`make_pipeline` 与参数网格存在不一致风险**：`l1_ratio` 仅在 `penalty='elasticnet'` 时有效，当前代码注释掉了 penalty 设置，可能导致搜索参数无效或报错（与 sklearn 版本相关）。
2. **`train_test_split_manual` 依赖未在 `__init__` 初始化的成员**（`self.rest_data/self.task_data`），当前主流程未调用该函数，但函数本身可用性不足。
3. **相对功率计算可读性**：先转 dB 再反变换等价于线性总功率，建议直接保留线性域避免误解。
4. **路径写法偏脚本式**：大量 `../data/...` 相对路径，若从非预期工作目录执行会失败。
5. **可视化模块较大**：`visualizer.py` 职责较多，可考虑按“基础图/SHAP图”拆分文件。

---

## 6. 如何快速运行（基于当前实现）

在仓库根目录执行：

```bash
python src/pipeline.py
```

前提：

- `data/raw/` 中有符合预期结构的 joblib 数据；
- 依赖安装齐全（numpy/scipy/scikit-learn/shap/seaborn/joblib/tqdm/matplotlib）；
- 运行目录与脚本中的相对路径匹配。

---

## 7. 总结

这是一个结构清晰、面向神经信号二分类的传统机器学习管线：

- 通过频带功率构建可解释特征；
- 使用跨被试 LOGO 评估泛化能力；
- 通过 SHAP 进一步增强可解释性。

整体具备较好的实验可复现性雏形，下一步建议优先修复模型参数网格与路径鲁棒性问题。
