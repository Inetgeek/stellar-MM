from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import h5py
import lmdb
import os
import numpy as np
import random

import torch
import torch.utils.data as data

import multiprocessing
import six
from torch.utils.data import Dataset, DataLoader


class TrainSet(Dataset):
    def __init__(self, X, Y, l):
        self.X, self.Y, self.label = X, Y, l

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.label[index]

    def __len__(self):
        return len(self.X)


def Dataloader(opt):
    path = opt['data_path']
    train_path = path + 'train'
    val_path = path + 'test'

    for i in range(opt['train_num']):

        num = i + 1
        str_num = str(num)  # 数字转化为字符串
        str_num = str_num.zfill(7)  # 字符串右对齐补0
        np_x = np.load(train_path + '/ifeature/P' + str_num + '.npy')
        np_y = np.load(train_path + '/tfeature/P' + str_num + '.npy')
        if num == 1:
            np_img = np_x
            np_text = np_y
        else:
            np_img = np.vstack((np_img, np_x))
            np_text = np.vstack((np_text, np_y))
    l = np.ones((opt['train_num']))
    l = torch.from_numpy(l)
    X_tensor = torch.from_numpy(np_img)
    Y_tensor = torch.from_numpy(np_text)
    mydataset = TrainSet(X_tensor, Y_tensor, l)
    train_loader = DataLoader(mydataset, batch_size=opt['BatchSize'], shuffle=True)
    # for step, (x, y, l) in enumerate(train_loader):
    # print(x,y,l)
    return train_loader


def main(opt):
    Dataloader(opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='./Data/')
    parser.add_argument('--train_num', type=int, default=43)
    parser.add_argument('--val_num', type=int, default=16)
    parser.add_argument('--BatchSize', type=int, default=16)
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
