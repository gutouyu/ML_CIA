import gc
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 1. 读取数据
path = "./data/"
train_file = "train3.csv"
test_file = "test3.csv"

trainDf = pd.read_csv(path + train_file)
# testDf = pd.read_csv(path + train_file, nrows=1000, skiprows=range(1, 10000))

pos_trainDf = trainDf[trainDf['target'] == 1]
neg_trainDf = trainDf[trainDf['target'] == 0].sample(n=20000, random_state=2018)
trainDf = pd.concat([pos_trainDf, neg_trainDf], axis=0).sample(frac=1.0, random_state=2018)
del pos_trainDf;
del neg_trainDf;
gc.collect();

print(trainDf.shape, trainDf['target'].mean())

trainDf, testDf, _, _ = train_test_split(trainDf, trainDf['target'], test_size=0.25, random_state=2018)

print(trainDf['target'].mean(), trainDf.shape)
print(testDf['target'].mean(), testDf.shape)

"""
一共59个特征，包括id， target
bin特征17个;cat特征14个;连续特征26个;
Code:

columns = trainDf.columns.tolist()
bin_feats = []
cat_feats = []
con_feats = []
for col in  columns:
    if 'bin' in col:
        bin_feats.append(col)
        continue
    if 'cat' in col:
        cat_feats.append(col)
        continue
    if 'id' != col and 'target' != col:
        con_feats.append(col)

print(len(bin_feats), bin_feats)
print(len(cat_feats), cat_feats)
print(len(con_feats), con_feats)
"""
bin_feats = ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin',
             'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_calc_15_bin',
             'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin']
cat_feats = ['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat',
             'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat',
             'ps_car_10_cat', 'ps_car_11_cat']
con_feats = ['ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_11',
             'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04',
             'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11',
             'ps_calc_12', 'ps_calc_13', 'ps_calc_14']

# 2. 特征处理
trainDf = trainDf.fillna(0)
testDf = testDf.fillna(0)

train_sz = trainDf.shape[0]
combineDf = pd.concat([trainDf, testDf], axis=0)
del trainDf
del testDf
gc.collect()

# 2.1 连续特征全部归一化
from sklearn.preprocessing import MinMaxScaler

for col in con_feats:
    scaler = MinMaxScaler()
    combineDf[col] = scaler.fit_transform(np.array(combineDf[col].values.tolist()).reshape(-1, 1))

# 2.2 离散特征一会直接进行embedding，暂时不用处理

# 3. 模型

# 3.1 初始化参数
NUM_FIELDS = len(bin_feats) + len(cat_feats)
EMBED_SZ = 10
field_sz = [0] * NUM_FIELDS
dropout_keep = [0.5, 0.5, 0.5]
hidden_layers = [512, 512, 1]
activation = [tf.nn.relu, tf.nn.relu, tf.nn.sigmoid]

# 隐藏层输入维度大小
num_pairs = NUM_FIELDS * (NUM_FIELDS - 1) / 2
lz_sz = num_pairs * EMBED_SZ
lp_sz = num_pairs
hidden_input_sz = num_pairs * EMBED_SZ + num_pairs # lz + lp

vars = []

## 初始化embedding层w  一个field一个w。 没有bias
layersz = hidden_input_sz
for i in range(NUM_FIELDS):
    vars['embed_%d' % i] = tf.Variable(tf.truncated_normal(shape=[layersz, hidden_layers[i]], mean=0.0, stddev=1e-3))
    layersz = hidden_layers[i]


# 3.2 搭积木-神经网络搭建

# 3.3 loss & opt

# 3.4 train & eval

