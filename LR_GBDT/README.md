# 介绍
针对CTR预估，测试LR + GBDT的方案效果. 

# 数据集

这里提供两份数据集，第一份比较好是CTR的，第二份也还凑合，之前在DeepFm中有用过。按理来说用第一个数据更好，但是压缩包大小为4G+ 有点大.
所以我采用的是第二个数据。感兴趣的同学，可以尝试下用第一个的数据进行试验。非常欢迎分享下实验结果~

## 1. kaggle CTR比赛
使用kaggle 2014年比赛 criteo-Display Advertising Challenge比赛的数据集。第一名的方案就是参考了Facebook的论文，使用GBDT进行特征转换，后面跟FFM

比赛地址： https://www.kaggle.com/c/criteo-display-ad-challenge/data
数据集下载：http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/

第一名方案参考
https://www.kaggle.com/c/criteo-display-ad-challenge/discussion/10555
PPT： https://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf


## 2. kaggle 比赛
kaggle上一个预测任务
https://www.kaggle.com/c/porto-seguro-safe-driver-prediction

