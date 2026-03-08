# 重构路线图（含训练与 SHAP 提速）

## Phase 1（本次已落地）
1. 修复关键逻辑 bug：`train_test_split_manual` 在 `min/fixed` 策略下训练集为空的问题。  
2. 训练提速：
   - 支持 `search_mode='random'`（`RandomizedSearchCV`）替代全量网格；
   - 暴露 `n_iter / n_jobs / inner_cv_splits` 参数，按资源调优。  
3. SHAP 提速：
   - 对 `StandardScaler + LogisticRegression` 的 Pipeline 走线性快速路径（`LinearExplainer`）；
   - 增加背景样本与 train/test 解释样本抽样参数，避免全量解释过慢。  

## Phase 2（建议 1~2 天内）
1. 引入统一配置：
   - 新增 `config.yaml`（数据路径、频段、CV、搜索策略、SHAP采样）。
2. 抽离训练入口：
   - 将 `pipeline.py` 拆分成 `train.py / explain.py / preprocess.py` 三个 CLI。
3. 提高可复现性：
   - 统一随机种子，记录每次训练配置和指标到 JSON。

## Phase 3（建议 3~5 天内）
1. 自动化评估：
   - 增加最小单元测试（split逻辑、特征维度、SHAP输出形状）。
2. 算法优化：
   - 在同等性能下比较 `SGDClassifier(loss='log_loss')` 与当前模型速度；
   - 对大样本实验尝试增量训练。
3. 可视化输出规范化：
   - 所有图表支持保存路径与无GUI模式。

## 参数建议（默认可用）
- 训练：`search_mode='random'`, `n_iter=10~20`, `inner_cv_splits=3`, `n_jobs=-1`。  
- SHAP：`background_size=50~100`, `train_max_samples=200~500`, `test_max_samples=200~500`, `prefer_linear_fastpath=True`。

