import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend

# Create input objerct which reads data from MNIST datasets. 
# Perform one-hot encoding to define the digit.
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Using Interactive session makes it the default sessions so we do not need to pass sess
sess = tf.InteractiveSession()

image_rows = 28
image_cols = 28

# Reshape the training and test images to 28 x 28 x 1
train_images = mnist.train.images.reshape(mnist.train.images.shape[0], image_rows, image_cols, 1)
test_images = mnist.test.images.reshape(mnist.test.images.shape[0], image_rows, image_cols, 1)

# Layer values
num_filters = 32
max_pool_size = (2, 2)
conv_kernel_size = (3, 3)
image_shape = (28, 28, 1)
num_classes = 10
drop_prob = 0.5

# Define the model type
model = Sequential()

# Define the layers in the neural network.
# Define the 1st convolution layer. We use border_mode= and input_shape only on the first layer.
# border_mode=value restricts convolution to only where the input and the filter fully overlap (ie. not partial overlap).
model.add(Convolution2D(num_filters, conv_kernel_size[0], 
                        conv_kernel_size[1], border_mode='valid',
                        input_shape=image_shape))

# Push through the activation
model.add(Activation('relu'))

# Take the result through the max pool
model.add(MaxPooling2D(pool_size=max_pool_size))

# 2nd Convolution layer
model.add(Convolution2D(num_filters, conv_kernel_size[0], conv_kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=max_pool_size))

# Fully connected layer
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

# Dropout to reduce overfitting
model.add(Dropout(drop_prob))

# Output layer
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Set the loss and measurement optmizer, and metric used to evaluate the loss
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Training settings
batch_size = 128
num_epoch = 2

# Fit the training data to the model. Nicely displays the time, loss, and validation accuracy on the test data
model.fit(train_images, 
          mnist.train.labels, 
          batch_size=batch_size,
          nb_epoch=num_epoch, 
          verbose=1,
          validation_data=(test_images, mnist.test.labels))
