from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from PIL import Image
import random

import cn_clip.clip
import cn_clip.clip as clip
import os
from cn_clip.clip import load_from_name, available_models
import numpy as np

import jsonlines
import json
import argparse

from dataloader import Dataloader
from Classifier import neural_net

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(opt):
    Clip, preprocess = load_from_name("ViT-L-14", device=device, download_root='./log')
    Clip.eval()
    loader = Dataloader(opt)
    net = neural_net()
    optimizer = torch.optim.Adam(net.parameters(), lr=opt['lr'])
    loss_func = torch.nn.CrossEntropyLoss()
    for i in range(opt['epoch']):
        for step, (i_feat, t_feat, Y) in enumerate(loader):
            feature = torch.cat([i_feat, t_feat], 1)
            Y_hat = net(feature)
            _ = np.zeros(Y.shape)
            _ = torch.from_numpy(_)
            _ = _.unsqueeze(-1)
            Y = Y.unsqueeze(-1)
            Y = torch.cat([Y, _], 1)
            loss = loss_func(Y, Y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if i == 99:
            print("y_hat")
            print(Y_hat)
            print("y")
            print(Y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--data_path', type=str, default='./Data/')
    parser.add_argument('--train_num', type=int, default=43)
    parser.add_argument('--val_num', type=int, default=16)

    # model
    parser.add_argument('--model', type=str, default='ViT-L-14')
    parser.add_argument('--log_path', type=str, default='./log')
    parser.add_argument('--BatchSize', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
