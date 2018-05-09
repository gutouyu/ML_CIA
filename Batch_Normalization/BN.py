import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


ACTIVATION = tf.nn.tanh
N_LAYERS = 7
N_HIDDEN_UNITS = 30

def fix_seed(seed=1):
    np.random.seed(seed)
    tf.set_random_seed(seed)

def plot_his(inputs, inputs_norm):
    # plot histogram for the inputs of every layer
    for j, all_inputs in enumerate([inputs, inputs_norm]):
        for i, input in enumerate(all_inputs):
            plt.subplot(2, len(all_inputs), j*len(all_inputs)+(i+1))
            plt.cla()
            if i == 0:
                the_range = (-7, 10)
            else:
                the_range = (-1, 1)
            plt.hist(input.ravel(), bins=15, range=the_range, color='#FF5733')
            plt.yticks(())
            if j == 1:
                plt.xticks(the_range)
            else:
                plt.xticks(())
            ax = plt.gca()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
        plt.title("%s normalizing" % ("Without" if j == 0 else "With"))
    plt.draw()
    plt.pause(0.01)

def built_net(xs, ys, norm):
    def add_layer(inputs, in_size, out_size, activation_function=None, norm=False):
        Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0.0, stddev=1.0))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

        # Full Connect
        Wx_plus_b = tf.matmul(inputs, Weights) + biases

        # Batch Normalization
        if norm:
            fc_mean, fc_var = tf.nn.moments(Wx_plus_b, axes=[0])
            scale = tf.Variable(tf.ones([out_size]))
            shift = tf.Variable(tf.zeros([out_size]))
            epsilon = 0.001

            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()

            # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
            # Wx_plus_b = Wx_plus_b * scale + shift
            Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)

        # Activation
        if activation_function == None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

        return outputs

    if norm:
        # BN for first layer
        fc_mean, fc_var = tf.nn.moments(xs, axes=[0])
        scale = tf.Variable(tf.ones([1]))
        shift = tf.Variable(tf.zeros([1]))
        epsilon = 0.001

        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)

        mean, var = mean_var_with_update()

        xs = tf.nn.batch_normalization(xs, mean, var, shift, scale, epsilon)

    # record.md inputs for every layer
    layers_inputs = [xs]

    # build hidden layer
    for layer_idx in range(N_LAYERS):
        layer_input = layers_inputs[layer_idx]
        in_size = layer_input.get_shape()[1].value

        output = add_layer(
            inputs              = layer_input,
            in_size             = in_size,
            out_size            = N_HIDDEN_UNITS,
            activation_function = ACTIVATION,
            norm                = norm
        )

        layers_inputs.append(output)

    # build output layer
    prediction = add_layer(layers_inputs[-1], N_HIDDEN_UNITS, 1, activation_function=None)

    cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss=cost)

    return train_op, cost, layers_inputs

# main progress
if __name__ == '__main__':

    fix_seed(2018)

    # make up data
    x_data = np.linspace(start=-7, stop=10, num=2500)[:, np.newaxis]
    np.random.shuffle(x_data)
    noise = np.random.normal(loc=0, scale=8, size=x_data.shape)
    y_data = np.square(x_data) - 5 + noise

    # plot input data
    plt.scatter(x=x_data, y=y_data)
    plt.show()

    # prepare tf
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])
    train_op,      cost,      layers_inputs      = built_net(xs, ys, norm=False)
    train_op_norm, cost_norm, layers_inputs_norm = built_net(xs, ys, norm=True)

    # init tf
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # record.md cost
    cost_his = []
    cost_his_norm = []

    plt.ion() # 打开交互模式
    plt.figure(figsize=(7,3))
    for i in range(250): #[0,249]
        print(i)
        if i % 50 == 0: #plot histgram
            all_inputs, all_inputs_norm = sess.run([layers_inputs, layers_inputs_norm], feed_dict={xs: x_data, ys:y_data})
            plot_his(all_inputs, all_inputs_norm)

        # train on batch
        sess.run([train_op, train_op_norm], feed_dict={xs:x_data, ys:y_data})

        # record.md cost
        cost_his.append(sess.run(cost, feed_dict={xs:x_data, ys:y_data}))
        cost_his_norm.append(sess.run(cost_norm, feed_dict={xs:x_data, ys:y_data}))

    plt.ioff()
    plt.figure()
    plt.plot(np.arange(len(cost_his)), np.array(cost_his), label='no BN')
    plt.plot(np.arange(len(cost_his)), np.array(cost_his_norm), label='BN')
    plt.legend()
    plt.show()




