import torch.utils.data.dataset as dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import h5py
import time
import math


class MatDataset(dataset.Dataset):  # 继承类：dataset.Dataset
    def __init__(self, train_mode, root_path, window_size):
        self.train_mode = train_mode    # 训练集：train_mode = 1    测试集：train_mode = 0
        # files_path下需要有“Train”文件夹和“Test”文件夹，分别作为训练集和测试集
        self.window_size = window_size
        self.files_path = root_path + '/Train' if train_mode == 1 else '/Test'
        self.files_list = os.listdir(self.files_path)   # 返回files_path及其子文件夹下所有文件名

    def __getitem__(self, idx):
        hdf = h5py.File(os.path.join(self.files_path, self.files_list[0]))
        with hdf as file:
            x0 = file['X'][idx*self.window_size:(idx+1)*self.window_size, 0:48]
            y0 = file['Y'][idx*self.window_size:(idx+1)*self.window_size, :]
        x = torch.tensor(np.array(x0)).float()
        y = torch.tensor(np.array(y0)).float()
        if torch.cuda.is_available():
            x = x.to('cuda')
        if torch.cuda.is_available():
            y = y.to('cuda')
        return x, y

    def __len__(self):
        mdata = h5py.File(os.path.join(self.files_path, self.files_list[0]))
        with mdata as file:
            length = math.floor(file['X'].shape[0]/self.window_size)
        return length


# 调试
if __name__ == '__main__':
    ROOT_PATH = "D:\\VN_DL_Dataset_WOT"
    dataset = MatDataset(train_mode=1, root_path=ROOT_PATH, window_size=40)
    start_time = time.time()
    train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    for data in train_dataloader:
        print(data)
        end_time = time.time()
        print(f'load time {end_time - start_time :.4f}')
        k = len(dataset)
        print('')
