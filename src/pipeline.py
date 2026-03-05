# 0. 导库
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from src.preprocess.preprocessor import Preprocessor
from src.featureExtract.feature_extract import FeatureExtractor


# 针对于uECoG数据
def reconstruct_datasets(raw_data):
    datas_list = raw_data['datasets']
    fs = raw_data['fs']
    label_mapping = {0:"non_task",1:"task"}

    rest_data = datas_list[0]
    task_data_list = []
    for i in range(1,len(datas_list)):
        task_data_list.append(datas_list[i])
    task_data = np.concatenate(task_data_list,axis=0)

    rest_data = rest_data[:,:128,:]
    task_data = task_data[:,:128,:]
    rest_labels = np.ones(rest_data.shape[0]) * 0
    task_labels = np.ones(task_data.shape[0]) * 1

    datasets = {0: (rest_data,rest_labels),1:(task_data,task_labels)}
    dataset = {"datasets":datasets,"label_mapping":label_mapping,"fs":fs}

    joblib.dump(dataset,os.path.join('../data/reconstruct_datasets',data_path.split("/")[3]))



if __name__ == '__main__':
    data_path = '../data/raw/test.pkl'
    raw_data = joblib.load(data_path)
    reconstruct_datasets(raw_data)

    dataset = joblib.load("../data/reconstruct_datasets/test.pkl")
    datasets = dataset['datasets']
    label_mapping = dataset['label_mapping']
    fs = dataset['fs']
    preprocessor = Preprocessor(dataset)
    preprocessed_data = preprocessor.preprocess([50, 100, 150], 1, 200, 2, 4)
    preprocessed_datasets = {0: preprocessed_data[0], 1: preprocessed_data[1]}

    fe = FeatureExtractor(fs)
    features_dict = fe.extract_features_and_labels(preprocessed_datasets)

    final_datasets = {
        "datasets":features_dict,
        "label_mapping":label_mapping,
        "fs":fs,
    }

    joblib.dump(final_datasets,os.path.join('../data/processed',data_path.split("/")[3]))
