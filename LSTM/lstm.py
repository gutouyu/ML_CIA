# SimpleRNN in numpy
import numpy as np

timesteps = 100
input_features = 32
output_features = 64

inputs = np.random.random(shape=(timesteps, input_features))

state_t = np.zeros(shape=(output_features,)) # init state

W = np.random.random(shape=(output_features, input_features))
U = np.random.random(shape=(output_features, output_features))
b = np.random.random(shape=(output_features,))

successive_outputs = []

for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b) #input_t state_t => output_t

    successive_outputs.append(output_t)

    state_t = output_t  # update state_t using output_t

final_outputs = np.concatenate(successive_outputs, axis=0) #get the final_output with shape=(timesteps, output_features)

# 5.1 SimpleRNN
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Embedding
from keras import datasets
from keras.preprocessing import sequence


max_features = 10000 # 我们只考虑最常用的10k词汇
maxlen = 100 # 每个评论我们只考虑100个单词

(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=max_features)
print(len(x_train), len(x_test))

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen) #长了就截断，短了就补0


model = Sequential()
model.add(Embedding())
