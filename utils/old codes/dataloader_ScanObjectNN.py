"""
ScanObjectNN download: http://103.24.77.34/scanobjectnn/h5_files.zip
"""

import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset


def load_scanobjectnn_data(partition):
    all_data = []
    all_label = []
    h5_name = 'dataset/ScanObjectNN/h5_files/main_split_nobg/' + partition + '_objectdataset.h5'
    f = h5py.File(h5_name, mode="r")
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


class ScanObjectNN(Dataset):
    def __init__(self, num_points, partition='training'):
        self.data, self.label = load_scanobjectnn_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'training':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ScanObjectNN(2048)
    test = ScanObjectNN(2048, 'test')
    print('Ali')
    c = 0
    for data, label in test:
        c = c + 1
        print(label)
        #print(data.shape)
       # print(label)
