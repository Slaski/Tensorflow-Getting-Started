# import tensorflow
import tensorflow as tf

sess = tf.Session()

# Verify we can print a String
hello = tf.constant("Hello Pluralsight from Tensorflow")
print(sess.run(hello))

# Perform some simple math
a = tf.constant(20)
b = tf.constant(22)
print('a + b = {0}'.format(sess.run(a + b)))
