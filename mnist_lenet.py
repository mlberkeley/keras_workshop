from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Activation, Convolution2D, Dense, Flatten, MaxPooling2D
from keras.utils import np_utils

import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

lenet = Sequential()
lenet.add(Convolution2D(6, 5, 5, input_shape=(28, 28, 1)))
lenet.add(MaxPooling2D(pool_size=(2, 2)))
lenet.add(Activation('relu'))
lenet.add(Convolution2D(16, 5, 5))
lenet.add(MaxPooling2D(pool_size=(2, 2)))
lenet.add(Activation('relu'))
lenet.add(Convolution2D(120, 3, 3))
lenet.add(Activation('relu'))
lenet.add(Flatten())
lenet.add(Dense(128))
lenet.add(Activation('relu'))
lenet.add(Dense(10))
lenet.add(Activation('softmax'))

lenet.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
lenet.fit(x_train, y_train, batch_size=64, nb_epoch=1, verbose=1, validation_data=(x_test, y_test))

result = lenet.evaluate(x_test, y_test, verbose=1)
print("Accuracy: ", result[1])
