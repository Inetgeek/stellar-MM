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


def main(opt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_from_name("ViT-L-14", device=device, download_root='../log')
    model.eval()

    dir_text = params['output_dir'] + './tfeature'
    dir_img = params['output_dir'] + './ifeature'
    img_path = opt['images_root']
    with open(opt['train_json'], "r", encoding='utf_8') as f:
        i = 0
        for item in jsonlines.Reader(f):
            print('processing {} pic'.format(i))
            img = img_path + '/' + item["id"] + '.' + item["ftype"]
            image = preprocess(Image.open(img)).unsqueeze(0).to(device)
            text = item['caption']
            text = cn_clip.clip.tokenize(text).to(device)
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            np.save(os.path.join(dir_text, str(item['id'])), text_features.data.cpu().float().numpy())
            np.save(os.path.join(dir_img, str(item['id'])), image_features.data.cpu().float().numpy())
            i += 1
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--train_json', type=str, default='../Datasets/COCO/train.jsonl')
    parser.add_argument('--test_json', type=str, default='../Datasets/COCO/val.jsonl')
    parser.add_argument('--output_dir', default='../Data', help='output h5 file')

    parser.add_argument('--images_root', default='../Datasets/COCO/train',
                        help='root location in which images are stored, to be prepended to file_path in input json')

    # model init
    parser.add_argument('--model', type=str, default='ViT-L-14')
    parser.add_argument('--BatchSize', type=int, default='256')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
