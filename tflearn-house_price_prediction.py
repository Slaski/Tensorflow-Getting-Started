#
#   house_price_prediction.py
#
#   This is a very simple prediction of house prices based on house size, implemented
#   in Tensorflow. This code is part of Pluralsight's course "Tensorflow: Getting Started"
#

import tensorflow as tf
import numpy as np
import math
import tflearn


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


# Set up Tensorflow palceholder that get updated as we descend down the gradient
tf_house_size = tf.placeholder(tf.float32, name='house_size')
tf_price = tf.placeholder(tf.float32, name='price')

# One value in, one value out. Names let us see these in TensorBoard.
input = tflearn.input_data(shape=[None], name='InputData')
linear = tflearn.layers.core.single_unit(input, activation='linear', name='Linear')

# Define the optimizer, the metric we try to optimize, and how we calculate loss.
reg = tflearn.regression(linear, 
                         optimizer='sgd', 
                         loss='mean_square', 
                         metric='R2', 
                         learning_rate=0.01, 
                         name='regression')

# Define the model
model = tflearn.DNN(reg)

# Train the mode with training data
model.fit(train_house_size_norm, train_price_norm, n_epoch=1000)
print('Training complete')

# Output W and b for the trained linear equation
print('Weights: W={0}, b={1}\n'.format(model.get_weights(linear.W), model.get_weights(linear.b)))

# Evaluate the accuracy
print(' Accuracy {0} '.format(model.evaluate(test_house_size_norm, test_house_price_norm)))