import argparse


def parse_opt():
    parser = argparse.ArgumentParser()

    # data input
    parser.add_argument('--train_json', type=str, default='./Datasets/COCO/train.jsonl')
    parser.add_argument('--test_json', type=str, default='./Datasets/COCO/val.json')
    parser.add_argument('--data_path', type=str, default='./Data/')
    parser.add_argument('--train_num', type=int, default=43)
    parser.add_argument('--val_num', type=int, default=16)

    # model init
    parser.add_argument('--model', type=str, default='ViT-L-14')
    parser.add_argument('--BatchSize', type=int, default='256')
