import os
import numpy as np
import joblib
from utils import Decomposer
import torch

def load(
        train=True,
        normalize=True,
        astorch = False
        ):
    dataset_root_path=r""
    if not os.path.exists(dataset_root_path):
        dataset_root_path=r""
        
    if train:
        data_train = joblib.load(dataset_root_path + "/train/data-multi-train-fixed.joblib")
        labels_train = joblib.load(dataset_root_path + "/train/labels-multi-train-fixed.joblib")
        
        if normalize:
            data_train = Decomposer(maxcurrent=500).normalize(data_train)
        

        data_train = np.array(data_train)
        labels_train = np.array(labels_train)

        if astorch:
            data_train = torch.Tensor(data_train).to(float)
            labels_train = torch.Tensor(labels_train).to(float)


        return data_train, labels_train
    
    else:
        data_test = joblib.load(dataset_root_path + "/test/data-multi-test-fixed.joblib")
        labels_test = joblib.load(dataset_root_path + "/test/labels-multi-test-fixed.joblib")

        if normalize:
            data_test = Decomposer(maxcurrent=500).normalize(data_test)
        data_test = np.array(data_test)
        labels_test = np.array(labels_test)

        if astorch:
            data_test = torch.Tensor(data_test).to(float)
            labels_test = torch.Tensor(labels_test).to(float)

        return data_test, labels_test

