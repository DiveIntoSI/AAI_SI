import os
import random
import numpy as np
import pandas as pd
import pickle
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.model_selection import StratifiedShuffleSplit


def save_train_val_txt(dataset_params):
    # 对train_folder数据集进行分层划分,txt存入data_folder下
    #  train_floder: 'data/train'
    raw_data_folder = dataset_params["raw_data_folder"]
    split_info_folder = dataset_params["split_info_folder"]
    K = dataset_params['n_spilt']
    if not os.path.exists(split_info_folder):
        os.makedirs(split_info_folder)
    # 保存K个分割txt文件的地址
    save_dir = os.path.join(split_info_folder, f'data_spilt_{K}')
    # 如果已经存在就直接返回
    if os.path.exists(save_dir):
        # pass
        return save_dir
    else:
        # 不存在则执行分割
        os.makedirs(save_dir)
    data = []
    for speaker in os.listdir(raw_data_folder):
        label = int(speaker[-3:])
        for speaker_one in os.listdir(os.path.join(raw_data_folder, speaker)):
            FileID = speaker_one
            data.append([FileID, label])

    df = pd.DataFrame(data, columns=['FileID', 'Label'])

    # 对train按照比例划分出训练集和验证集
    split = StratifiedShuffleSplit(n_splits=K, test_size=0.3, random_state=111)
    for i, (train_index, test_index) in enumerate(split.split(df, df["Label"])):
        df.loc[train_index.tolist()].to_csv(os.path.join(save_dir, f"train_info{i}.txt"), index=False)
        df.loc[test_index.tolist()].to_csv(os.path.join(save_dir, f"val_info{i}.txt"), index=False)
    return save_dir



class MyDataSet(Dataset):
    def __init__(self, info_txt, add_noise, data_folder):
        self.info_txt = info_txt
        self.add_noise = add_noise
        self.data_folder = data_folder
        self.data_info = pd.read_csv(info_txt)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        file_id = self.data_info.loc[index]['FileID']
        if self.add_noise:
            file_id += '_noised'
        item = os.path.join(self.data_folder, file_id+'.pickle')
        data = None
        with open(item, "rb") as f:
            load_dict = pickle.load(f)
            data = load_dict["LogMel_Features"]
        return data, self.data_info.loc[index]['Label']


