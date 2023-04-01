import argparse


def parse_opt():
    parser = argparse.ArgumentParser()

    # data input
    parser.add_argument('--train_json', type=str, default='./Datasets/COCO/train.jsonl')
    parser.add_argument('--test_json', type=str, default='./Datasets/COCO/val.json')

    # model init
    parser.add_argument('--model', type=str, default='ViT-L-14')
    parser.add_argument('--BatchSize', type=int, default='256')
