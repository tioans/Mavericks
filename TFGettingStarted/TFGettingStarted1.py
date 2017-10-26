import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

sess = tf.Session()
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3):", sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b   # shortcut for tf.add(a,b)

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# variables
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x+b

# variables are only initialized when you call:
init = tf.global_variables_initializer()
sess.run(init)
# variable are still not initialized until you call sess.run(init)
# init is a handle to the TensorFLow sub-graph that initialized
# all the global variables

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))