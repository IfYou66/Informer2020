import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.tools import StandardScaler  # 请确保你使用的是 Informer 中的 StandardScaler
from utils.timefeatures import time_features  # 请确保提供了该函数


class Dataset_Favorita(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='train.csv',
                 target='sales', scale=True, inverse=False, timeenc=0, freq='D', cols=None):
        """
        size: [seq_len, label_len, pred_len]
        flag: 'train' / 'val' / 'test'
        features: 'M' 表示多变量，'S' 表示单变量（仅 target）
        """
        # 默认窗口长度（可根据数据情况调整）
        if size is None:
            self.seq_len = 30  # 历史观察 30 天
            self.label_len = 15  # Decoder 部分预留 15 天已知信息（warm-up）
            self.pred_len = 15  # 预测未来 15 天
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        # 填充缺失值，防止后续计算出 NaN（例如 lag、rolling 特征可能会产生 NaN）
        df_raw.fillna(0, inplace=True)

        # 按比例划分：70% 训练，20% 验证，10% 测试
        n = len(df_raw)
        num_train = int(n * 0.7)
        num_val = int(n * 0.2)
        num_test = n - num_train - num_val
        border1s = [0, num_train - self.seq_len, n - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, n]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 自动构造特征列表：
        # 如果没有传入特定的 cols，则自动使用除 'date' 和目标列之外的所有列
        if self.cols is None:
            cols = list(df_raw.columns)
            if 'date' in cols:
                cols.remove('date')
            if self.target in cols:
                cols.remove(self.target)
        else:
            cols = self.cols.copy()
            if 'date' in cols:
                cols.remove('date')
            if self.target in cols:
                cols.remove(self.target)

        # 拼接特征列和目标列，注意此时 df_data 中不包含日期列
        df_data = df_raw[cols + [self.target]]

        # 缩放：仅对数值部分进行标准化（缩放器以训练集数据拟合）
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 构造时间特征：利用 utils/timefeatures.py 中的 time_features 函数
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    # def __read_data__(self):
    #     self.scaler = StandardScaler()
    #     df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
    #     df_raw['date'] = pd.to_datetime(df_raw['date'])
    #
    #     # 按比例划分：70% 训练，20% 验证，10% 测试
    #     n = len(df_raw)
    #     num_train = int(n * 0.7)
    #     num_val = int(n * 0.2)
    #     num_test = n - num_train - num_val
    #     border1s = [0, num_train - self.seq_len, n - num_test - self.seq_len]
    #     border2s = [num_train, num_train + num_val, n]
    #     border1 = border1s[self.set_type]
    #     border2 = border2s[self.set_type]
    #
    #     # 选择特征：若 features=='M'，则取除 date 外所有列；若 'S' 则仅 target
    #     if self.features == 'M':
    #         cols_data = df_raw.columns.drop('date')
    #         df_data = df_raw[cols_data]
    #     elif self.features == 'S':
    #         df_data = df_raw[[self.target]]
    #     else:
    #         df_data = df_raw[self.cols + [self.target]]
    #
    #     # 缩放：仅对数值部分进行标准化（缩放器以训练集数据拟合）
    #     if self.scale:
    #         train_data = df_data[border1s[0]:border2s[0]]
    #         self.scaler.fit(train_data.values)
    #         data = self.scaler.transform(df_data.values)
    #     else:
    #         data = df_data.values
    #
    #     # 构造时间特征：利用 utils/timefeatures.py 中的 time_features 函数
    #     df_stamp = df_raw[['date']][border1:border2]
    #     df_stamp['date'] = pd.to_datetime(df_stamp['date'])
    #     data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
    #
    #     self.data_x = data[border1:border2]
    #     if self.inverse:
    #         self.data_y = df_data.values[border1:border2]
    #     else:
    #         self.data_y = data[border1:border2]
    #     self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin + self.label_len],
                                    self.data_y[r_begin + self.label_len:r_end]], axis=0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return torch.tensor(seq_x, dtype=torch.float), \
            torch.tensor(seq_y, dtype=torch.float), \
            torch.tensor(seq_x_mark, dtype=torch.float), \
            torch.tensor(seq_y_mark, dtype=torch.float)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
