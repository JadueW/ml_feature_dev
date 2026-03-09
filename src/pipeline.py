# 0. 导库
import os
import joblib
from tqdm import tqdm
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from src.preprocess.preprocessor import Preprocessor
from src.featureExtract.feature_extract import FeatureExtractor
from src.models.ml_feature_model import FeatureModel
from src.visualize.visualizer import Visualizer
from src.shap_analysis.shap_analysis import ShapAnalyzer


# 针对于uECoG数据
def reconstruct_datasets(raw_data,raw_data_path):
    datas_list = raw_data['datasets']
    fs = raw_data['fs']
    label_mapping = {0: "non_task", 1: "task"}

    rest_data = datas_list[0]
    task_data_list = []
    for i in range(1, len(datas_list)):
        task_data_list.append(datas_list[i])
    task_data = np.concatenate(task_data_list, axis=0)

    rest_data = rest_data[:, :128, :]
    task_data = task_data[:, :128, :]
    rest_labels = np.ones(rest_data.shape[0]) * 0
    task_labels = np.ones(task_data.shape[0]) * 1

    datasets = {0: (rest_data, rest_labels), 1: (task_data, task_labels)}
    dataset = {"datasets": datasets, "label_mapping": label_mapping, "fs": fs}

    joblib.dump(dataset, os.path.join('../data/reconstruct_datasets', raw_data_path.split("/")[3]))


if __name__ == '__main__':

    # 1. 文件夹目录定义
    RAW_DATASET_PATH = '../data/raw/'
    RECONSTRUCTED_DATASET_PATH = '../data/reconstruct_datasets/'
    PROCESSED_DATASET_PATH = '../data/processed/'
    MODEL_SAVE_PATH = '../models/'

    os.makedirs(MODEL_SAVE_PATH,exist_ok=True)
    os.makedirs(RECONSTRUCTED_DATASET_PATH,exist_ok=True)
    os.makedirs(PROCESSED_DATASET_PATH,exist_ok=True)

    RAW_DATA_LIST = [f for f in os.listdir('../data/raw/')]
    for f in RAW_DATA_LIST:

        # 2. 数据格式重组
        raw_data_path = os.path.join(os.path.join(RAW_DATASET_PATH, f))
        raw_data = joblib.load(raw_data_path)
        reconstruct_datasets(raw_data,raw_data_path)

        reconstruct_data_path = os.path.join(RECONSTRUCTED_DATASET_PATH, f)
        reconstruct_data = joblib.load(reconstruct_data_path)
        datasets = reconstruct_data['datasets']
        label_mapping = reconstruct_data['label_mapping']
        fs = reconstruct_data['fs']

        # 3. 数据预处理和特征提取
        preprocessor = Preprocessor(reconstruct_data)
        preprocessed_data = preprocessor.preprocess([50, 100, 150], 1, 200, 2, 4)
        preprocessed_datasets = {0: preprocessed_data[0], 1: preprocessed_data[1]}

        fe = FeatureExtractor(fs)
        features_dict = fe.extract_features_and_labels(preprocessed_datasets)
        final_datasets = {
            "datasets":features_dict,
            "label_mapping":label_mapping,
            "fs":fs,
        }
        joblib.dump(final_datasets,os.path.join(PROCESSED_DATASET_PATH,reconstruct_data_path.split("/")[3]))

    # 4. 加载处理后的数据
    processed_files = [os.path.join(PROCESSED_DATASET_PATH, f) for f in os.listdir(PROCESSED_DATASET_PATH)]
    X_list, y_list, group_list = [], [], []
    fs = None
    print("加载数据...")
    for subj_idx, file_path in enumerate(tqdm(processed_files)):
        data = joblib.load(file_path)
        rest_data = data["datasets"][0]
        task_data = data["datasets"][1]

        rest_label = np.zeros(len(rest_data))
        task_label = np.ones(len(task_data))

        X_subj = np.vstack([rest_data, task_data])
        y_subj = np.hstack([rest_label, task_label])
        group_subj = np.full(len(X_subj), subj_idx)

        X_list.append(X_subj)
        y_list.append(y_subj)
        group_list.append(group_subj)

        if fs is None:
            fs = data["fs"]

    X_all = np.vstack(X_list)
    y_all = np.hstack(y_list)
    groups_all = np.hstack(group_list)
    print(f"\n总样本: {X_all.shape[0]}, 被试数: {len(np.unique(groups_all))}")

    # 5. 基于网格搜索和LeaveOneGroupOut的跨被试模型训练和交叉验证
    fm = FeatureModel(X_all, y_all, groups_all, fs)
    results = fm.cross_validate_logo(n_jobs=-1)
    best_model = None

    # 6. 取其中一个被试的结果进行可视化验证
    last_subj = len(results) - 1
    if last_subj in results:
        Visualizer.plot_auc(
            results[last_subj]['test']['fpr'],
            results[last_subj]['test']['tpr'],
            results[last_subj]['test']['auc']
        )
        Visualizer.plot_confusion_matrix(results[last_subj]['test']['cm'])


    # 7. 模型保存
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    model_name = f"cross_subjects_model_{date_str}.pkl"
    joblib.dump(best_model, os.path.join(MODEL_SAVE_PATH,model_name))
    print(f"模型已保存至：{MODEL_SAVE_PATH}")


    # # 可视化auc 和混淆矩阵
    # Visualizer.plot_auc(fpr, tpr, auc)
    # Visualizer.plot_confusion_matrix(cm)

    # # SHAP分析
    # analyzer = ShapAnalyzer(best_model, X_train, X_test,n_channels=128, n_bands=6, n_types=2)
    # analyzer.compute_shap(background_size=100)
    #
    # # 可视化SHAP
    # analyzer.plot_channel_importance()
    # analyzer.plot_band_importance(separate_types=True)
    # analyzer.plot_type_importance()
    # analyzer.plot_channel_band_heatmap(type_idx=None)  # 合并类型
    # analyzer.plot_channel_band_heatmap(type_idx=0)  # abs
    # analyzer.plot_channel_band_heatmap(type_idx=1)  # rel
    # analyzer.plot_direction_summary(top_k_channels=10)
    # analyzer.plot_summary(max_display=30)
