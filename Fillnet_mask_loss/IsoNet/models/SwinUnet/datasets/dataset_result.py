import os
from torch.utils.data import Dataset
import numpy as np
import mrcfile

class tomograms_result(Dataset): #获取目标值和输入值对应
    def __init__(self, base_dir,split, transform=None):
        #获取train_x,train_y,test_x,test_y 里面的数据地址是歌二维数组。
        dirs_tomake = ['train_x', 'train_y', 'test_x', 'test_y']
        path_all = []
        for d in dirs_tomake:
            p = '{}\\{}\\'.format(base_dir, d)
            path_all.append(sorted([p + f for f in os.listdir(p)]))

        self.path_all = path_all #所有的训练集和测试集包括输入和目标值
        self.transform = transform  # using transform in torch!
        self.split = split
        #获取train 和 test 的数据长度
        if split == 'train':
            self.len = len(path_all[0])
        else:
            self.len = len(path_all[2])
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        #这里获取的是我要一个idx 然后获取相应的键值对
        if self.split == "train": #这里作用获得两个3维矩阵一个输入一个目标
            image = np.array([mrcfile.open(self.path_all[0][idx], permissive=True).data]).squeeze()
            label = np.array([mrcfile.open(self.path_all[1][idx], permissive=True).data]).squeeze()
        else:
            image = np.array([mrcfile.open(self.path_all[2][idx], permissive=True).data]).squeeze()
            label = np.array([mrcfile.open(self.path_all[3][idx], permissive=True).data]).squeeze()
        sample = {'image': image, 'label': label}
        if self.transform:# 对图像进行增强
            pass
        return sample