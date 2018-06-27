# Criteo CTR数据

## kaggle
https://www.kaggle.com/c/criteo-display-ad-challenge

## Data fields
Label - Target variable that indicates if an ad was clicked (1) or not (0).
I1-I13 - A total of 13 columns of integer features (mostly count features).
C1-C26 - A total of 26 columns of categorical features. The values of these features have been hashed onto 32 bits for anonymization purposes. 

## 数据集划分
* train 90%的训练数据
* val 10%的训练数据
* test 全部的测试数据

## mini文件
分别取训练集、测试集、验证集前10000条数据组成。原始的数据太大了，大家可以跑我上传的这三个小文件。每个文件都包含10000条样本，特征都处理好了。
想要完整数据的私信我吧


# Reference
1. 特征处理参考：https://github.com/PaddlePaddle/models/blob/develop/deep_fm/preprocess.py
