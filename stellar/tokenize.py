#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""
 @author: Colyn
 @project: utils.py
 @devtool: PyCharm
 @date: 2023/3/30
 @file: tokenize.py
"""

import os
import re
import shutil
import thulac
import jsonlines


def jsonl_write(jsonl: dict, path: str):
    """
    (追加)逐行写入jsonl文件API
    :param jsonl: 需要写入的jsons，为dict
    :param path: 需要写入的jsonl文件的路径
    :return: None
    """
    with jsonlines.open(path, mode='a') as w:
        w.write(jsonl)


def jsonl_read(path: str):
    """
    逐行读取jsonl文件中json的API
    :param path: 需要读取的jsonl文件路径
    :return: json
    """
    assert os.path.exists(path), f"The {path} is not exist!"
    with jsonlines.open(path, mode='r') as r:
        for row in r:
            yield row


def get_tokenize(string: str, query: tuple):
    """
    中文分词API
    :param string: 需要分词的文本
    :param query: 分词规则
    :return: list of tokens
    """
    thu = thulac.thulac()
    text = thu.cut(string, text=False)
    # print(text)
    return [_[0] for _ in text if _[1] in query]


def cpy_file(src_dir: str, aim_dir: str):
    """
    拷贝文件API, 若文件存在则不进行拷贝
    :param src_dir: 源文件目录
    :param aim_dir: 目标文件目录
    :return: None
    """
    assert os.path.exists(src_dir), f"The {src_dir} is not exist!"
    index = aim_dir.rfind(re.findall('/', aim_dir)[-1])
    os.makedirs(os.path.join(aim_dir[:index]), exist_ok=True)
    if not os.path.exists(aim_dir):
        shutil.copyfile(src_dir, aim_dir)
    else:
        pass


# def gen_template(string: str, pattern: str) -> str:
#     thu = thulac.thulac()
#     text = thu.cut(string, text=False)
