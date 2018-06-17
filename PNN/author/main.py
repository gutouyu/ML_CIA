from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

if sys.version[0] == '2':
    pass
else:
    pass

import numpy as np
from sklearn.metrics import roc_auc_score

import progressbar

import os
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not p in sys.path:
    sys.path.append(p)

from PNN.author import utils
from PNN.author.models import LR, FM, PNN1, PNN2, FNN, CCPM, DeepFM

train_file = '../data/train.txt'
test_file = '../data/test.txt'

input_dim = utils.INPUT_DIM

train_data = utils.read_data(train_file)
# train_data = pkl.load(open('../data/train.yx.pkl', 'rb'))
train_data = utils.shuffle(train_data)
test_data = utils.read_data(test_file)
# test_data = pkl.load(open('../data/test.yx.pkl', 'rb'))
# pkl.dump(train_data, open('../data/train.yx.pkl', 'wb'))
# pkl.dump(test_data, open('../data/test.yx.pkl', 'wb'))

if train_data[1].ndim > 1:
    print('label must be 1-dim')
    exit(0)
print('read finish')
print('train data size:', train_data[0].shape)
print('test data size:', test_data[0].shape)

train_size = train_data[0].shape[0]
test_size = test_data[0].shape[0]
num_feas = len(utils.FIELD_SIZES)

min_round = 1
num_round = 200
early_stop_round = 5
batch_size = 1024

field_sizes = utils.FIELD_SIZES
field_offsets = utils.FIELD_OFFSETS

algo = 'pnn2'

if algo in {'fnn', 'ccpm', 'pnn1', 'pnn2', 'deepfm'}:
    train_data = utils.split_data(train_data)
    test_data = utils.split_data(test_data)
    tmp = []
    for x in field_sizes:
        if x > 0:
            tmp.append(x)
    field_sizes = tmp
    print('remove empty fields', field_sizes)

if algo == 'lr':
    lr_params = {
        'input_dim': input_dim,
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'l2_weight': 0,
        'random_seed': 0
    }
    print(lr_params)
    model = LR(**lr_params)
elif algo == 'fm':
    fm_params = {
        'input_dim': input_dim,
        'factor_order': 10,
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'l2_w': 0,
        'l2_v': 0,
    }
    print(fm_params)
    model = FM(**fm_params)
elif algo == 'fnn':
    fnn_params = {
        'field_sizes': field_sizes,
        'embed_size': 10,
        'layer_sizes': [500, 1],
        'layer_acts': ['relu', None],
        'drop_out': [0, 0],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'embed_l2': 0,
        'layer_l2': [0, 0],
        'random_seed': 0
    }
    print(fnn_params)
    model = FNN(**fnn_params)
elif algo == 'deepfm':
    deepfm_params = {
        'field_sizes': field_sizes,
        'embed_size': 10,
        'layer_sizes': [500, 1],
        'layer_acts': ['relu', None],
        'drop_out': [0, 0],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'embed_l2': 0,
        'layer_l2': [0, 0],
        'random_seed': 0
    }
    print(deepfm_params)
    model = DeepFM(**deepfm_params)
elif algo == 'ccpm':
    ccpm_params = {
        'field_sizes': field_sizes,
        'embed_size': 10,
        'filter_sizes': [5, 3],
        'layer_acts': ['relu'],
        'drop_out': [0],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'random_seed': 0
    }
    print(ccpm_params)
    model = CCPM(**ccpm_params)
elif algo == 'pnn1':
    pnn1_params = {
        'field_sizes': field_sizes,
        'embed_size': 10,
        'layer_sizes': [500, 1],
        'layer_acts': ['relu', None],
        'drop_out': [0, 0],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'embed_l2': 0,
        'layer_l2': [0, 0],
        'random_seed': 0
    }
    print(pnn1_params)
    model = PNN1(**pnn1_params)
elif algo == 'pnn2':
    pnn2_params = {
        'field_sizes': field_sizes,
        'embed_size': 10,
        'layer_sizes': [500, 1],
        'layer_acts': ['relu', None],
        'drop_out': [0, 0],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'embed_l2': 0,
        'layer_l2': [0., 0.],
        'random_seed': 0,
        'layer_norm': True,
    }
    print(pnn2_params)
    model = PNN2(**pnn2_params)


def train(model):
    history_score = []
    for i in range(num_round):
        fetches = [model.optimizer, model.loss]
        if batch_size > 0:
            ls = []
            bar = progressbar.ProgressBar()
            print('[%d]\ttraining...' % i)
            for j in bar(range(int(train_size / batch_size + 1))):
                X_i, y_i = utils.slice(train_data, j * batch_size, batch_size)
                _, l = model.run(fetches, X_i, y_i)
                ls.append(l)
        elif batch_size == -1:
            X_i, y_i = utils.slice(train_data)
            _, l = model.run(fetches, X_i, y_i)
            ls = [l]
        train_preds = []
        print('[%d]\tevaluating...' % i)
        bar = progressbar.ProgressBar()
        for j in bar(range(int(train_size / 10000 + 1))):
            X_i, _ = utils.slice(train_data, j * 10000, 10000)
            preds = model.run(model.y_prob, X_i, mode='test')
            train_preds.extend(preds)
        test_preds = []
        bar = progressbar.ProgressBar()
        for j in bar(range(int(test_size / 10000 + 1))):
            X_i, _ = utils.slice(test_data, j * 10000, 10000)
            preds = model.run(model.y_prob, X_i, mode='test')
            test_preds.extend(preds)
        train_score = roc_auc_score(train_data[1], train_preds)
        test_score = roc_auc_score(test_data[1], test_preds)
        print('[%d]\tloss (with l2 norm):%f\ttrain-auc: %f\teval-auc: %f' % (i, np.mean(ls), train_score, test_score))
        history_score.append(test_score)
        if i > min_round and i > early_stop_round:
            if np.argmax(history_score) == i - early_stop_round and history_score[-1] - history_score[
                        -1 * early_stop_round] < 1e-5:
                print('early stop\nbest iteration:\n[%d]\teval-auc: %f' % (
                    np.argmax(history_score), np.max(history_score)))
                break

train(model)