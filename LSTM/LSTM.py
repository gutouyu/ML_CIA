def plot_acc_and_loss(history):
    from matplotlib import pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validtion acc')
    plt.legend()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

# 5.2 LSTM
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras import datasets
from keras.preprocessing import sequence


max_features = 10000 # 我们只考虑最常用的10k词汇
maxlen = 500 # 每个评论我们只考虑100个单词

(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=max_features)
print(len(x_train), len(x_test))

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen) #长了就截断，短了就补0


model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc']
)

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)

plot_acc_and_loss(history)
rets = model.evaluate(x_test, y_test)
print(rets)
