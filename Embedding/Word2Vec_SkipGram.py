from tensorflow import set_random_seed
set_random_seed(1)
from numpy.random import seed
seed(1)

from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow import keras

"""
实现简单版本的Skip-gram模型
"""


"""
1. 准备数据
"""
corpus = "The quick brown fox jumped over the lazy dog"


# 全局变量记录下来当前遍历的数量
data_index = 0

"""
num_skips: 每个中心词生成多少个训练样本；num_skips <= 2 * skip_window
skip_window: 考虑多远的上下文； 值为1，则只考虑前后各一个词，共两个词
num_skips 必须能被batch_size整除，以保证同一个中心词对应的所有样本都在一个batch中
span: 表示当前操作的所有单词长度，包括中心词和上下文，span = 2 * skip_window + 1
"""
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray()
