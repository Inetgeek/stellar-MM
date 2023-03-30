# stellar-MM
 stellar for 4th China AI Competition(2023)

## 数据集结构

> 数据集文件目录，此处以COCO为例：

```
- Datasets: # 数据集目录
	- COCO:	# COCO数据集，不同类型的数据集文件结构与命名规则一致
		- train:	# 训练集
			- P0000001.PNG
			...
		- test:		# 测试集 
			- T000001.PNG
			...
		- val:		# 验证集
			- V0000001.PNG
			...
		- captions:	#字幕文件
			- train.jsonl
			- test.jsonl
			- val.jsonl
			
	...	# 其他数据集
```



-----------------------

### 图片

> 图片命名采用编号形式，分别存储在不同的文件目录下。 

其中：

1.**训练集**（`./Datasets/{DATASET}/train/`）

> 图片命名均以<span style="font-color:red">标识字母+7位长度的自增数字+图片格式</span>组成，其中数字从1开始自增。**图片格式最好是JPG、PNG、JPEG中的一种。**

- **正样本**的标识字母为`P`，例如：`P0253181.JPG`。
- **负样本**的标识字母为`N`，例如：`N0253181.JPG`。



2.**测试集**（`./Datasets/{DATASET}/test/`）

> 图片命名均以<span style="font-color:red">标识字母+7位长度的自增数字+图片格式</span>**图片格式最好是JPG、PNG、JPEG中的一种。**

- 图片标识字母为`T`，例如：`T0028341.PNG`。



3.验证集（`./Datasets/{DATASET}/val/`）

> 图片命名均以<span style="font-color:red">标识字母+7位长度的自增数字+图片格式</span>**图片格式最好是JPG、PNG、JPEG中的一种。**

- 图片标识字母为`V`，例如：`V0000316.JPEG`



--------------------------

### 字幕文件

> 字幕文件均采用`jsonl`文件形成存储，即文件的每一行都是一个图片的字幕描述的`json`文件，方便流式读取和处理。

如：`train.jsonl`

```jsonl
{"type":"train_p","id":"P0000001","ftype":"PNG","caption":["A dog is running in the sky.","In the sunny sky, a dog is running."],"tags": ["dog","sky"]}
{"type":"train_p","id":"P0000001","ftype":"PNG","caption":["A dog is running in the sky.","In the sunny sky, a dog is running."],"tags": ["dog","sky"]}
{"type":"train_p","id":"P0000001","ftype":"PNG","caption":["A dog is running in the sky.","In the sunny sky, a dog is running."],"tags": ["dog","sky"]}
{"type":"train_p","id":"P0000001","ftype":"PNG","caption":["A dog is running in the sky.","In the sunny sky, a dog is running."],"tags": ["dog","sky"]}
{"type":"train_p","id":"P0000001","ftype":"PNG","caption":["A dog is running in the sky.","In the sunny sky, a dog is running."],"tags": ["dog","sky"]}
...
```

json格式

```
{
    "type": "val",	//图片类型
    "id": "P0000001",	//图片ID，即图片名称
    "ftype": "PNG",	//图片格式，支持PNG、JPG及JPEG
    "caption": ["A dog is running in the sky.","In the sunny sky, a dog is running."],	//字幕列表
    "tags": ["dog","sky"]	//图像里的关键标签
}
```


其中：

- **type**: 图片类型，有`train_p`、`train_n`、`test`、`val`四种类型
- **id**：图片ID，即图片名称（见[图片](#图片)部分）。
- **ftype**：图片格式，支持PNG、JPG及JPEG。
- **caption**：字幕列表
- **tags**：图像里的关键标签



## 编码规则

> 采用模块集成方式。

1.自己编写的模块文件名均**小写**，如`log.py`，该模块用于打印日志时使用。

2.自己编写模块测试无误后请在`./stellar/__init__.py`中的`__all__ = []`中添加自己的模块名。假设自己编写了`coco_cn.py`模块，则添加为`___all__ = ['log', 'coco_cn']`。

比赛所用的代码集成到python包名为`stellar`的文件结构里，文件结构如下：

```
- stellar:
	- __init__.py	# 包初始化模块
	- log.py		# 打印日志的模块
	- coco_cn.py	# 自编码的模块
	
```

其中，`__init__.py`文件内容如下：

```python
from . import *

__all__ = ['log'] # 新增的模块需要在此处添加
```

