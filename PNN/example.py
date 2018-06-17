import tensorflow as tf
import numpy as np


k = 3

e1 = tf.constant([1,2,3], shape=[1,k])
e2 = tf.constant([4,5,6], shape=[1,k])

kernel = tf.constant(np.arange(k * k), shape=[k, k], dtype=tf.int32)

tmp = tf.matmul(e1, kernel, transpose_a=True)
ret = tf.matmul(tmp, e2)

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)

print(sess.run(ret))

