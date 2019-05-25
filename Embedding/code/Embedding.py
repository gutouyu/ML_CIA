import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()

embedding = tf.Variable(np.identity(5, dtype=np.int32))
input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])
input_embeddings = tf.nn.embedding_lookup(embedding, input_ids)



feat_index = [
    [0,          3,           7,           8,             9],
    [1,           4,           7,           8,             9],
    [2,           3,           7,           8,             9],
    [0,           5,           7,           8,             9],
    [2,           6,           7,           8,             9],
]

feat_val = [
[1.0,         1.0,         3.1,         2.2,             0],
[1.0,         1.0,         2.1,         3.1,             0],
[1.0,         1.0,         1.0,         3.4,             0],
[1.0,         1.0,         2.1,         1.6,             0],
[1.0,         1.0,         0.5,         1.8,             0]
]

#origin

# print(sess.run(embedding))
# print("="*50)
#
# feat_index = [
#     [1,2,3,4,3],
#     [1,2,3,1,4],
#     [2,3,1,2,4],
# ]
#
# # print(sess.run(input_embeddings, feed_dict={input_ids:[1,2,3,0,3,2,1,4]}))
# print(sess.run(input_embeddings, feed_dict={input_ids:feat_index}))

embedding2 = tf.Variable(np.identity(n=10,dtype=np.int32)[:,:8])
tf_input_index = tf.placeholder(dtype=tf.int32, shape=[None,None])
tf_input_value = tf.placeholder(dtype=tf.float32, shape=[None,None])

input_embedding2 = tf.nn.embedding_lookup(embedding2, tf_input_index)
feat_value = tf.reshape(tf_input_value, shape=[-1, 5, 1])
# input_embedding3 = tf.matmul(tf.cast(input_embedding2, tf.float32), feat_val)


sess.run(tf.global_variables_initializer())
print(sess.run(input_embedding2, feed_dict={tf_input_index:feat_index, tf_input_value:feat_val}))
print(sess.run(feat_value, feed_dict={tf_input_index:feat_index, tf_input_value:feat_val}))
# print(sess.run(input_embedding3, feed_dict={tf_input_index:feat_index, tf_input_value:feat_val}))