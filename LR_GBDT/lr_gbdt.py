import gc
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
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
del pos_trainDf; del neg_trainDf; gc.collect();

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
bin_feats = ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin']
cat_feats = ['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_10_cat', 'ps_car_11_cat']
con_feats = ['ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_11', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14']

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
    combineDf[col] = scaler.fit_transform(np.array(combineDf[col].values.tolist()).reshape(-1,1))

# 2.2 离散特征one-hot
for col in bin_feats + cat_feats:
    onehotret = pd.get_dummies(combineDf[col], prefix=col)
    combineDf = pd.concat([combineDf, onehotret], axis=1)


# 3. 训练模型
label = 'target'
onehot_feats = [col for col in combineDf.columns if col not in ['id', 'target'] + con_feats + cat_feats + bin_feats]
train = combineDf[:train_sz]
test = combineDf[train_sz:]
print("Train.shape: {0}, Test.shape: {0}".format(train.shape, test.shape))
del combineDf

# 3.1 LR模型
lr_feats = con_feats + onehot_feats
lr = LogisticRegression(penalty='l2', C=1)
lr.fit(train[lr_feats], train[label].values)


def do_model_metric(y_true, y_pred, y_pred_prob):
    print("Predict 1 percent: {0}".format(np.mean(y_pred)))
    print("Label 1 percent: {0}".format(train[label].mean()))
    from sklearn.metrics import roc_auc_score,accuracy_score
    print("AUC: {0:.3}".format(roc_auc_score(y_true=y_true, y_score=y_pred_prob[:,1])))
    print("Accuracy: {0}".format(accuracy_score(y_true=y_true, y_pred=y_pred)))
print("Train............")
do_model_metric(y_true=train[label], y_pred=lr.predict(train[lr_feats]), y_pred_prob=lr.predict_proba(train[lr_feats]))

print("\n\n")
print("Test.............")
do_model_metric(y_true=test[label], y_pred=lr.predict(test[lr_feats]), y_pred_prob=lr.predict_proba(test[lr_feats]))


# 3.2 GBDT
lgb_feats = con_feats + cat_feats + bin_feats
categorical_feature_list = cat_feats + bin_feats

import lightgbm as lgb
lgb_params ={
    'objective':'binary',
    'boosting_type': 'gbdt',
    'metric':'auc',
    'learning_rate': 0.01,
    'num_leaves': 5,
    'max_depth': 4,
    'min_data_in_leaf': 100,
    'bagging_fraction': 0.8,
    'feature_fraction':0.8,
    'bagging_freq':10,
    'lambda_l1':0.2,
    'lambda_l2':0.2,
    'scale_pos_weight':1,
}

lgbtrain = lgb.Dataset(train[lgb_feats].values, label=train[label].values,
                          feature_name=lgb_feats,
                          categorical_feature=categorical_feature_list
                          )
lgbvalid = lgb.Dataset(test[lgb_feats].values, label=test[label].values,
                          feature_name=lgb_feats,
                          categorical_feature=categorical_feature_list
                          )

evals_results = {}
print('train')
lgb_model = lgb.train(lgb_params,
                 lgbtrain,
                 valid_sets=lgbvalid,
                 evals_result=evals_results,
                 num_boost_round=1000,
                 early_stopping_rounds=60,
                 verbose_eval=50,
                 categorical_feature=categorical_feature_list,
                 )


# 3.3 LR + GBDT
train_sz = train.shape[0]
combineDf = pd.concat([train, test], axis=0, ignore_index=True)


#得到叶节点编号 Feature Transformation
gbdt_feats_vals = lgb_model.predict(combineDf[lgb_feats], pred_leaf=True)
gbdt_columns = ["gbdt_leaf_indices_" + str(i) for i in range(0, gbdt_feats_vals.shape[1])]

combineDf = pd.concat([combineDf, pd.DataFrame(data=gbdt_feats_vals, index=range(0, gbdt_feats_vals.shape[0]),columns=gbdt_columns)], axis=1)

# onehotencoder(gbdt_feats)
origin_columns = combineDf.columns
for col in gbdt_columns:
    combineDf = pd.concat([combineDf, pd.get_dummies(combineDf[col], prefix=col)],axis=1)
gbdt_onehot_feats = [col for col in combineDf.columns if col not in origin_columns]

# 恢复train, test
train = combineDf[:train_sz]
test = combineDf[train_sz:]
del combineDf; gc.collect();

lr_gbdt_feats = lr_feats + gbdt_onehot_feats

lr_gbdt_model = LogisticRegression(penalty='l2', C=1)
lr_gbdt_model.fit(train[lr_gbdt_feats], train[label])

print("Train................")
do_model_metric(y_true=train[label], y_pred=lr_gbdt_model.predict(train[lr_gbdt_feats]), y_pred_prob=lr_gbdt_model.predict_proba(train[lr_gbdt_feats]))

print("Test..................")
do_model_metric(y_true=test[label], y_pred=lr_gbdt_model.predict(test[lr_gbdt_feats]), y_pred_prob=lr_gbdt_model.predict_proba(test[lr_gbdt_feats]))

