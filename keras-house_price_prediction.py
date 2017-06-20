#
#   house_price_prediction.py
#
#   This is a very simple prediction of house prices based on house size, implemented
#   in Tensorflow. This code is part of Pluralsight's course "Tensorflow: Getting Started"
#

import tensorflow as tf
import numpy as np
import math

from keras.models import Sequential
from keras.layers.core import Dense, Activation


# Generate some house sizes between 1000 and 3500 (typical sq ft of houses)
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)


# Generate house prices from the sizes with a random noise added
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)


# You need to normalize values to prevent under/overflows.
def normalize(array):
    return (array - array.mean()) / array.std()

# Define number of training samples, 0.7 = 70%. We can take the first 70% since values are randomized
num_train_samples = math.floor(num_house * 0.7)


# Define the training data
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asarray(house_price[:num_train_samples])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)


# Define the test data
test_house_size = np.asarray(house_size[num_train_samples:])
test_house_price = np.asarray(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)


# Define the neural network for doing Linear Regression
model = Sequential()
model.add(Dense(1, input_shape=(1,), init='uniform', activation='linear'))
model.compile(loss='mean_squared_error', optimizer='sgd') # Loss and optimizer


# Fit/train the model
model.fit(train_house_size_norm, train_price_norm, nb_epoch=300)


# Evaluate the model. Note: the fit cost values will be different because we did not use NN in original.
score = model.evaluate(test_house_size_norm, test_house_price_norm)
print('\nLoss on test: {}'.format(score))
