#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""
 @author: Colyn
 @project: utils.py
 @devtool: PyCharm
 @date: 2023/3/30
 @file: main.py
"""

from stellar.tokenize import *

if __name__ == '__main__':

    noun = ('n', 'np', 'ns', 'ni', 'nz')  # 构造名词查询表
    temp = ('n', 'np', 'ns', 'ni', 'nz', 'mq')  # 构造模板查询表

    caption_dir = './coco-cn_ext.icap2020.txt'

    with open(caption_dir, 'r', encoding='utf-8') as f:
        json = {'ftype': 'JPG'}
        t_cnt, v_cnt = 0, 0
        for line in f.readlines():
            num = ''
            l = line.strip('\n').split('\t')
            # str(nums // 10) + str(nums % 10)
            # val ['COCO_val2014_000000217153#0', '一个男人和一个女人穿着军装玩手机。'] COCO_val2014_000000217153
            # print(l[0][5:-19], l, l[0][:-2])
            tokens = get_tokenize(l[1], query=noun)
            # ['男人', '女人', '军装', '手机']
            # print(tokens)
            # val
            if l[0][5:-19] == 'val':
                v_cnt += 1
                num += 'V'
                num += "{:07d}".format(v_cnt)
                # print("n:", num)  # n: V00000001
                json['id'] = num
                json['type'] = 'val'
                json['caption'] = l[1]
                json['tags'] = tokens
                os.makedirs(os.path.join('./Datasets/COCO/val'), exist_ok=True)
                jsonl_write(json, './Datasets/COCO/val.jsonl')
                cpy_file(src_dir=f'./val2014/{l[0][:-2]}.jpg', aim_dir=f'./Datasets/COCO/val/{num}.jpg')
            # train
            if l[0][5:-19] == 'train':
                t_cnt += 1
                num += 'P'
                num += "{:07d}".format(v_cnt)
                # print("n:", num)  # n: V00000001
                json['id'] = num
                json['type'] = 'train_p'
                json['caption'] = l[1]
                json['tags'] = tokens
                os.makedirs(os.path.join('./Datasets/COCO/train'), exist_ok=True)
                jsonl_write(json, './Datasets/COCO/train.jsonl')
                cpy_file(src_dir=f'./train2014/{l[0][:-2]}.jpg', aim_dir=f'./Datasets/COCO/train/{num}.jpg')
