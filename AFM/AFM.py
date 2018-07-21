#!/usr/bin/env python
#coding=utf-8
"""
TensorFlow Implementation of AFM
"""

import tensorflow as tf
import tensorflow.contrib as contrib
#0 1:0.05 2:0.006633 3:0.05 4:0 5:0.021594 6:0.008 7:0.15 8:0.04 9:0.362 10:0.1 11:0.2 12:0 13:0.04 15:1 555:1 1078:1 17797:1 26190:1 26341:1 28570:1 35361:1 35613:1 35984:1 48424:1 51364:1 64053:1 65964:1 66206:1 71628:1 84088:1 84119:1 86889:1 88280:1 88283:1
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
    """Build Model function f(x) for Estimator."""

    #------hyper parameters------
    field_size = params['field_size']
    feature_size = params['feature_size']
    embedding_size = params['embedding_size']
    l2_reg = params['l2_reg']
    learning_rate = params['learning_rate']

    dropout = params['dropout']
    attention_factor = params['attention_factor']

    #------build weights------
    Global_Bias = tf.get_variable("bias", shape=[1], initializer=tf.constant_initializer(0.0))
    Feat_Wgts = tf.get_variable("linear", shape=[feature_size], initializer=tf.glorot_normal_initializer())
    Feat_Emb = tf.get_variable("emb", shape=[feature_size, embedding_size], initializer=tf.glorot_normal_initializer())

    #------build feature------
    feat_ids = features['feat_ids']
    feat_vals = features['feat_vals']
    feat_ids = tf.reshape(feat_ids, shape=[-1, field_size])
    feat_vals = tf.reshape(feat_vals, shape=[-1, field_size]) # None * F

    #------build f(x)------

    # FM部分: sum(wx)
    with tf.variable_scope("Linear-part"):
        feat_wgts = tf.nn.embedding_lookup(Feat_Wgts, feat_ids) # None * F * 1
        y_linear = tf.reduce_sum(tf.multiply(feat_wgts, feat_vals), 1)

    #Deep部分
    with tf.variable_scope("Embedding_Layer"):
        embeddings = tf.nn.embedding_lookup(Feat_Emb, feat_ids) # None * F * K
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1]) # None * F * 1
        embeddings = tf.multiply(embeddings, feat_vals) # None * F * K


    with tf.variable_scope("Pair-wise_Interaction_Layer"):
        num_interactions = field_size * (field_size - 1) / 2
        element_wise_product_list = []
        for i in range(0, field_size):
            for j in range(i + 1, field_size):
                element_wise_product_list.append(tf.multiply(embeddings[:, i, :], embeddings[:, j, :]))
        element_wise_product_list = tf.stack(element_wise_product_list) # (F*(F-1)/2) * None * K stack拼接矩阵
        element_wise_product_list = tf.transpose(element_wise_product_list, perm=[1,0,2]) # None * (F(F-1)/2) * K

    # 得到Attention Score
    with tf.variable_scope("Attention_Netowrk"):

        deep_inputs = tf.reshape(element_wise_product_list, shape=[-1, embedding_size]) # (None*F(F-1)/2) * K

        deep_inputs = contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=attention_factor, activation_fn=tf.nn.relu, \
                                             weights_regularizer=contrib.layers.l2_regularizer(l2_reg), scope="attention_net_mlp")

        aij = contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity, \
                                             weights_regularizer=contrib.layers.l2_regularizer(l2_reg), scope="attention_net_out") # (None*F(F-1)/2) * 1

        # 得到attention score之后，使用softmax进行规范化
        aij = tf.reshape(aij, shape=[-1, int(num_interactions), 1])
        aij_softmax = tf.nn.softmax(aij, dim=1, name="attention_net_softout") # None * num_interactions

        # TODO: 为什么要对attention score进行dropout那？? 这里不是很懂
        if mode == tf.estimator.ModeKeys.TRAIN:
            aij_softmax = tf.nn.dropout(aij_softmax, keep_prob=dropout[0])

    with tf.variable_scope("Attention-based_Pooling_Layer"):
        deep_inputs = tf.multiply(element_wise_product_list, aij_softmax) # None * (F(F-1)/2) * K
        deep_inputs = tf.reduce_sum(deep_inputs, axis=1) # None * K Pooling操作

        # Attention-based Pooling Layer的输出也要经过Dropout
        if mode == tf.estimator.ModeKeys.TRAIN:
            deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[1])

        # 该层的输出是一个K维度的向量

    with tf.variable_scope("Prediction_Layer"):
        # 直接跟上输出单元
        deep_inputs = contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity, \
                                             weights_regularizer=contrib.layers.l2_regularizer(l2_reg), scope="afm_out") # None * 1
        y_deep = tf.reshape(deep_inputs, shape=[-1]) # None

    with tf.variable_scope("AFM_overall"):
        y_bias = Global_Bias * tf.ones_like(y_deep, dtype=tf.float32)
        y = y_bias + y_linear + y_deep
        pred = tf.nn.sigmoid(y)

    # set predictions
    predictions = {"prob": pred}
    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    #------build loss------
    loss =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels))+ l2_reg * tf.nn.l2_loss(Feat_Wgts) + l2_reg * tf.nn.l2_loss(Feat_Emb)
    log_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels))


    # Provide an estimator spec for `ModeKeys.EVAL`
    eval_metric_ops = {
        # "logloss": tf.losses.log_loss(pred, labels, weights=1.0, scope=None, epsilon=1e-07,loss_collection=tf.GraphKeys.LOSSES, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS),
        "auc": tf.metrics.auc(labels, pred),
    }

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops)


    #------build optimizer------
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN`
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=log_loss, # 只打印pure log_loss，但是训练依旧按照整个的loss来训练
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
    'dropout':[0.5,0.5], # 分别表示Attention Network， Attention-based Pooling Layer的输出的dropout
    'attention_factor': 256, # attention network是一个one-layer MLP, 表示其神经元个数
    'train_or_debug': "debug",
}


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

