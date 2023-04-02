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


def feature_norm(image_features, text_features):
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    return image_features, text_features


def get_negative(model, i_feat, t_feat):
    _i_feat = i_feat.clone()
    _t_feat = t_feat.clone()
    label = torch.empty((i_feat.shape[0], 2))
    logits_per_image, logits_per_text = model.my_get_similarity(_i_feat, _t_feat)
    index = logits_per_image.softmax(dim=-1)
    for i in range(index.shape[0]):
        maxIndex = -1
        maxP = 0
        for j in range(index.shape[1]):
            if i == j:
                continue
            if index[i][j] > maxP:
                maxIndex = j
                maxP = index[i][j]
        _t_feat[i] = t_feat[maxIndex]
        label[i][1] = 1.
    return i_feat.clone(), _t_feat, label


def main(opt):
    Clip, preprocess = load_from_name("ViT-L-14", device=device, download_root='./log')
    Clip.eval()
    loader = Dataloader(opt)
    net = neural_net()
    optimizer = torch.optim.SGD(net.parameters(), lr=opt['lr'])
    loss_func = torch.nn.CrossEntropyLoss()
    for i in range(opt['epoch']):
        for step, (i_feat, t_feat, Y) in enumerate(loader):
            i_feat, t_feat = feature_norm(i_feat, t_feat)
            _i_feat, _t_feat, _label = get_negative(Clip, i_feat, t_feat)
            feature = torch.cat([i_feat, t_feat], 1)
            _feature = torch.cat([_i_feat, _t_feat], 1)
            input_feature = torch.cat((feature, _feature), dim=0)
            label = torch.empty((i_feat.shape[0], 2))
            for _ in range(i_feat.shape[0]):
                label[_][0] = 1
            input_label = torch.cat((label, _label), dim=0)
            label_hat = net(input_feature)
            loss = loss_func(label_hat, input_label)
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--data_path', type=str, default='./Data/')
    parser.add_argument('--train_num', type=int, default=43)
    parser.add_argument('--val_num', type=int, default=16)

    # model
    parser.add_argument('--model', type=str, default='ViT-L-14')
    parser.add_argument('--log_path', type=str, default='./log')
    parser.add_argument('--BatchSize', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
