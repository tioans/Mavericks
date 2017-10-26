

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

# We can create two floating point Tensors node1 and node2
# (tips: const. are initialized and value can never change when you call tf.const.)
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0, dtype=tf.float32) #also tf.float32 implicitly

# The following creates a Session object and then invokes its run method 
# to run enough of the computational graph to evaluate node1 and node2.
sess = tf.Session()
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3)", sess.run(node3))

# A graph can be parameterized to accept external inputs, known as placeholders. 
# A placeholder is a promise to provide a value later.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + provides a shortcut for tf.add(a, b)

# We can evaluate this graph with multiple inputs by using the feed_dict argument 
# to the run method to feed concrete values to the placeholders:
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# making the computational graph more complex by adding another operation
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# Variables allow us to add trainable parameters to a graph. 
# They are constructed with a type and initial value:
# (tips: variables or not initialezed when you call tf.Variables and can change)
m = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = m * x + b

# To initialize all the variables in a TensorFlow program, 
# you must explicitly call a special operation as follows:
# (tips: init is a handle to the TensorFlow sub-graph that 
# initializes all the global variables)
init = tf.global_variables_initializer()
sess.run(init)

# Since x is a placeholder, we can evaluate linear_model for 
# several values of x simultaneously as follows:
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

# To evaluate the model on training data, we need a y placeholder 
# to provide the desired values, and we need to write a loss function.
# Standard loss model for linear regression.
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


# We could improve this manually by reassigning the values of m and b to 
# the perfect values of -1 and 1.
# Initialized variables by tf.Variables can be changed using tf.assign.

fixm = tf.assign(m, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixm, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# We guessed the "perfect" values of W and b, but the whole point of 
# machine learning is to find the correct model parameters automatically. 
# We will show how to accomplish this in the next section.


























