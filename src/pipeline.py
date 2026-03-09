# 0. 导库
import os
import joblib
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
    RAW_DATASET_PATH = '../data/raw/'
    RECONSTRUCTED_DATASET_PATH = '../data/reconstruct_datasets/'
    PROCESSED_DATASET_PATH = '../data/processed/'

    RAW_DATA_LIST = [f for f in os.listdir('../data/raw/')]
    for f in RAW_DATA_LIST:
        raw_data_path = os.path.join(os.path.join(RAW_DATASET_PATH, f))
        raw_data = joblib.load(raw_data_path)
        reconstruct_datasets(raw_data,raw_data_path)

        reconstruct_data_path = os.path.join(RECONSTRUCTED_DATASET_PATH, f)
        reconstruct_data = joblib.load(reconstruct_data_path)
        datasets = reconstruct_data['datasets']
        label_mapping = reconstruct_data['label_mapping']
        fs = reconstruct_data['fs']
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

    # data_path = '../data/processed/test2.pkl'
    # dataset = joblib.load(data_path)
    # datasets = dataset['datasets']
    # label_mapping = dataset['label_mapping']
    # fs = dataset['fs']
    #
    # fm = FeatureModel(datasets, label_mapping, fs)
    # X_train, y_train, X_test, y_test = fm.train_test_split_manual(strategy='ratio')
    #
    # # 获取最佳结果和所有结果
    # best_eval_result, all_eval_results = fm.train_eval_splits(X_train, y_train, X_test, y_test, n_splits=1)
    # best_model = best_eval_result['best_model']
    # fpr = best_eval_result['fpr']
    # tpr = best_eval_result['tpr']
    # auc = best_eval_result['auc']
    # cm = best_eval_result['cm']
    #
    # model_save_path = "./ml_decoder_model.pkl"
    # joblib.dump(best_model, model_save_path)
    # print(best_eval_result)
    # print(f"模型已保存至：{model_save_path}")
    #
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
