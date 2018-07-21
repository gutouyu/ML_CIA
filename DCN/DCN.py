#!/usr/bin/env python
#coding=utf-8
"""
TensorFlow Implementation of DCN
"""

import tensorflow as tf
import tensorflow.contrib as contrib
def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
    print('Parsing', filenames)
    def decode_libsvm(line):
        columns = tf.string_split([line], ' ')
        labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
        splits = tf.string_split(columns.values[1:], ':')
        id_vals = tf.reshape(splits.values,splits.dense_shape)
        feat_ids, feat_vals = tf.split(id_vals,num_or_size_splits=2,axis=1)
        feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
        feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
        return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=10).prefetch(1000)

    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size) # Batch size to use

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None, is_training=True, reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z
def model_fn(features, labels, mode, params):
    """Bulid Model function f(x) for Estimator."""
    #------hyperparameters----
    field_size = params["field_size"]
    feature_size = params["feature_size"]
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]
    deep_layers  = map(int, params["deep_layers"].split(','))
    cross_layers = params["cross_layers"]
    dropout = map(float, params["dropout"].split(','))

    #------bulid weights------
    Cross_B = tf.get_variable(name='cross_b', shape=[cross_layers, field_size*embedding_size], initializer=tf.glorot_normal_initializer())
    Cross_W = tf.get_variable(name='cross_w', shape=[cross_layers, field_size*embedding_size], initializer=tf.glorot_normal_initializer())
    Feat_Emb = tf.get_variable(name='emb', shape=[feature_size, embedding_size], initializer=tf.glorot_normal_initializer())

    #------build feaure-------
    feat_ids  = features['feat_ids']
    feat_ids = tf.reshape(feat_ids,shape=[-1,field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals,shape=[-1,field_size])

    #------build f(x)------
    with tf.variable_scope("Embedding-layer"):
        embeddings = tf.nn.embedding_lookup(Feat_Emb, feat_ids) 		    # None * F * K
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])
        embeddings = tf.multiply(embeddings, feat_vals) 				    # None * F * K
        x0 = tf.reshape(embeddings,shape=[-1,field_size*embedding_size])    # None * (F*K)

    with tf.variable_scope("Cross-Network"):
        xl = x0
        for l in range(cross_layers):
            wl = tf.reshape(Cross_W[l],shape=[-1,1])                        # (F*K) * 1
            xlw = tf.matmul(xl, wl)                                         # None * 1
            xl = x0 * xlw + xl + Cross_B[l]                                 # None * (F*K) broadcast

    with tf.variable_scope("Deep-Network"):
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_phase = True
        else:
            train_phase = False

        x_deep = x0
        for i in range(len(deep_layers)):
            x_deep = tf.contrib.layers.fully_connected(inputs=x_deep, num_outputs=deep_layers[i], \
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='mlp%d' % i)
            if mode == tf.estimator.ModeKeys.TRAIN:
                x_deep = tf.nn.dropout(x_deep, keep_prob=dropout[i])                              #Apply Dropout after all BN layers and set dropout=0.8(drop_ratio=0.2)

    with tf.variable_scope("DCN-out"):
        x_stack = tf.concat([xl, x_deep], 1)	# None * ( F*K+ deep_layers[i])
        y = tf.contrib.layers.fully_connected(inputs=x_stack, num_outputs=1, activation_fn=tf.identity, weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='out_layer')
        y = tf.reshape(y,shape=[-1])
        pred = tf.sigmoid(y)

    predictions={"prob": pred}
    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)

    #------bulid loss------
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)) + \
        l2_reg * tf.nn.l2_loss(Cross_B) +  l2_reg * tf.nn.l2_loss(Cross_W) + l2_reg * tf.nn.l2_loss(Feat_Emb)

    # Provide an estimator spec for `ModeKeys.EVAL`
    eval_metric_ops = {
        "auc": tf.metrics.auc(labels, pred)
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metric_ops)

    #------bulid optimizer------
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op)


#---------------------------------------------------------------------------------------------------------------
#------------------------------------------ Main Function ------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
# 日志级别
tf.logging.set_verbosity(tf.logging.INFO)

print("构建分类器......")
model_params = {
    'field_size': 13 + 26,
    'feature_size': 117581,
    'embedding_size': 64,
    'l2_reg': 0.0005,
    'learning_rate': 0.001,
    'dropout':[0.5,0.5],
    'train_or_debug': "debug",
}


field_size = params["field_size"]
feature_size = params["feature_size"]
embedding_size = params["embedding_size"]
l2_reg = params["l2_reg"]
learning_rate = params["learning_rate"]
deep_layers  = map(int, params["deep_layers"].split(','))
cross_layers = params["cross_layers"]
dropout = map(float, params["dropout"].split(','))


log_steps = 10
if model_params['train_or_debug'] == "train":
    train_file = '../EveryTestInOne/criteo/tr.libsvm'
    test_file = '../EveryTestInOne/criteo/te.libsvm'
    val_file = '../EveryTestInOne/criteo/va.libsvm'
else:
    # prefetch the head for 10000 rows, just for debug
    # train_file = '../EveryTestInOne/criteo/tr.mini.libsvm'
    # test_file = '../EveryTestInOne/criteo/te.mini.libsvm'
    # val_file = '../EveryTestInOne/criteo/va.mini.libsvm'
    train_file = './data/tr.mini.libsvm'
    test_file = './data/te.mini.libsvm'
    val_file = './data/va.mini.libsvm'

print("初始化......")
config = tf.estimator.RunConfig().replace(
    session_config=tf.ConfigProto(device_count={'GPU': 0, 'CPU': 10}),
    log_step_count_steps=log_steps, save_summary_steps=log_steps)
classifier = tf.estimator.Estimator(model_fn=model_fn,model_dir='./model_save', params=model_params, config=config)  # Path to where checkpoints etc are stored

print("训练......")
classifier.train(input_fn=lambda: input_fn(train_file, 256, 1, True))

print("评估......")
evaluate_result = classifier.evaluate(input_fn=lambda: input_fn(val_file, 256, 1, False))
for key in evaluate_result:
    tf.logging.info("{}, was: {}".format(key, evaluate_result[key]))

evaluate_result = classifier.evaluate(input_fn=lambda: input_fn(train_file, 256, 1, False))
for key in evaluate_result:
    tf.logging.info("{}, was: {}".format(key, evaluate_result[key]))

print("预测......")
predict_results = classifier.predict(input_fn=lambda: input_fn(test_file, 256, 1, False))
for prediction in predict_results:
    tf.logging.info("{}".format(prediction["prob"]))
    break

