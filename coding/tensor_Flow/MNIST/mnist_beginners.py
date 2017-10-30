import tensorflow as tf
import ssl
import os

# stopping the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# overrides the default function for context creation with the function to create an unverified context.
ssl._create_default_https_context = ssl._create_unverified_context
# download and read in the data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# placeholder 2D tensor of floating-point numbers, with a shape [None, 784]
x = tf.placeholder(tf.float32, [None, 784])
# variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# model, one line!
y = tf.nn.softmax(tf.matmul(x, W) + b)

# placeholder for our "one_hot" vector (label)
y_ = tf.placeholder(tf.float32, [None, 10])
# loss/cost/cross entropy function, this why is more stable.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# launching model
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# training our model
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Evaluating our model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))