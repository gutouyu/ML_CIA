from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding

print(tf.__version__)


"""
1. 准备训练数据
"""
# Context|Target
corpus =['eat|apple','eat|orange','eat|rice','drink|juice','drink|milk','drink|water',
         'orange|juice','apple|juice','rice|milk','milk|drink','water|drink','juice|drink']
corpus_king_queen = ['king|kindom','queen|kindom','king|palace','queen|palace',
                     'king|royal','queen|royal','king|George','queen|Mary','man|rice',
                     'woman|rice','man|farmer','woman|farmer','man|house','woman|house',
                     'man|George','woman|Mary']
corpus_king_queen_symbol = ['king|a','queen|a','king|b','queen|b','king|c','queen|c','king|x',
                            'queen|y','man|d','woman|d','man|e','woman|e','man|f','woman|f',
                            'man|x','woman|y']

train_data = [sample.split('|')[0] for sample in corpus_king_queen_symbol]
train_label = [sample.split('|')[1] for sample in corpus_king_queen_symbol]

print(train_data)
print(train_label)

vocabulary = (list(set(train_data) | set(train_label)))
vocabulary.sort()
print(vocabulary)
print(len(vocabulary))

one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(np.reshape(vocabulary, (-1, 1)))

X = one_hot_encoder.transform(np.reshape(train_data, (-1, 1))).toarray()
y = one_hot_encoder.transform(np.reshape(train_label, (-1, 1))).toarray()


"""
2. 构建模型

输入是X，y
"""
print(X)
print(y)

N =  5
V = len(vocabulary)

inputs = Input(shape=(V, ))
x = Dense(N, activation='linear', use_bias=False)(inputs)
predictions = Dense(V, activation='softmax', use_bias=False)(x)



model = Model(inputs=inputs, outputs=predictions)
model.summary()

"""
3. 训练模型
"""
model.compile(optimizer=keras.optimizers.Adagrad(0.07),
              loss='categorical_crossentropy',
              metrics=['accuracy', 'mse'])
model.fit(X, y,
          batch_size=1,
          epochs=1000
          )

"""
4. 验证/可视化结果
"""
weights = model.get_weights()
embeddings = np.array(weights[0])
assert (embeddings.shape == (V, N))
word_vec = dict((word, vector) for word, vector in zip(vocabulary, embeddings))

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(embeddings)
print(X_pca)

fig, ax = plt.subplots()
for i in range(len(X_pca)):
    team = X_pca[i]
    ax.scatter(team[0], team[1])
    ax.annotate(vocabulary[i], (team[0], team[1]))
plt.show()