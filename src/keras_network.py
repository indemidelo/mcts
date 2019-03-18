import tensorflow as tf
from tensorflow import keras


def ResidualBlock(input, regularizer):
    # Convolutional layer #1
    conv1 = keras.layers.Conv2D(
        filters=236,
        kernel_size=[4, 4],
        padding='same',
        strides=1,
        kernel_regularizer=regularizer
    )(input)
    # Batch normalization #1
    batchnorm1 = keras.layers.BatchNormalization()(conv1)
    # ReLU #1
    relu1 = keras.layers.Activation('relu')(batchnorm1)
    # Convolutional layer #2
    conv2 = keras.layers.Conv2D(
        filters=236,
        kernel_size=[4, 4],
        padding='same',
        strides=1,
        kernel_regularizer=regularizer
    )(relu1)
    # Batch normalization #2
    batchnorm2 = keras.layers.BatchNormalization()(conv2)
    # Skip connection
    skip = tf.add(batchnorm2, input)
    # ReLU #2
    relu2 = keras.layers.Activation('relu')(skip)
    return relu2


def ResidualTower(input, regularizer, n_blocks):
    res = input
    for _ in range(n_blocks):
        res = ResidualBlock(res, regularizer)
    return res


def AlphaGo19Net(inputs, pi, z, beta, n_res_blocks, learning_rate):
    # Regularizer
    regularizer = keras.regularizers.l2(0.1)

    ### Body
    # Convolutional layer #1
    conv1 = keras.layers.Conv2D(
        filters=236,
        kernel_size=[4, 4],
        padding='same',
        strides=1,
        kernel_regularizer=regularizer
    )(inputs)
    # Batch normalization layer #1
    batchnorm1 = keras.layers.BatchNormalization()(conv1)
    # ReLU layer #1
    relu2 = keras.layers.Activation('relu')(batchnorm1)
    # Tower of residual blocks
    tower = ResidualTower(relu2, regularizer, n_res_blocks)

    ### Policy head
    # Convolutional layer #3
    conv3 = keras.layers.Conv2D(
        filters=2,
        kernel_size=[1, 1],
        padding='same',
        strides=1,
        kernel_regularizer=regularizer
    )(tower)
    # Batch normalization layer #4
    batchnorm4 = keras.layers.BatchNormalization()(conv3)
    # ReLU layer #4
    relu4 = keras.layers.Activation('relu')(batchnorm4)
    # Fully connected layer #1
    with tf.name_scope('PolicyHead'):
        relu4 = keras.layers.Flatten()(relu4)
        fc1 = keras.layers.Dense(
            units=7,
            kernel_regularizer=regularizer
        )(relu4)
        pred_policy = keras.layers.Activation('softmax')(fc1)

    ### Value Head
    # Convolutional layer #4
    conv4 = keras.layers.Conv2D(
        filters=1,
        kernel_size=[1, 1],
        strides=1,
        kernel_regularizer=regularizer
    )(tower)
    # Batch normalization #5
    batchnorm5 = keras.layers.BatchNormalization()(conv4)
    # ReLU #5
    relu5 = keras.layers.Activation('relu')(batchnorm5)
    # Fully connected layer #2
    relu5 = keras.layers.Flatten()(relu5)
    fc2 = keras.layers.Dense(
        units=256,
        kernel_regularizer=regularizer
    )(relu5)
    # ReLU #6
    relu6 = keras.layers.Activation('relu')(fc2)
    # Fully connected layer #3
    fc_output = keras.layers.Dense(
        units=1,
        kernel_regularizer=regularizer
    )(relu6)
    # Tanh activator
    pred_value = keras.layers.Activation('tanh')(fc_output)

    # Collect all regularization losses
    regularization_losses = tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES)

    ### Loss
    with tf.name_scope('Loss'):
        loss_value = tf.reshape(
            tf.squared_difference(z, pred_value), (-1,))
        loss_policy = tf.reduce_sum(
            tf.multiply(pi, tf.math.log(1e-6 + pred_policy)), axis=1)
        regularization = beta * tf.reduce_sum(regularization_losses)
        loss = tf.reduce_sum(loss_value - loss_policy) + regularization

    # todo add momentum = 0.9

    # Configure optimizer
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(loss)

    # Accuracy
    # with tf.name_scope('Accuracy'):
    # todo fix this accuracy
    # acc_policy = tf.reduce_mean(pred_policy - pi)
    # acc_value = tf.reduce_mean(pred_value - z)
    acc_policy = tf.reduce_sum(loss_policy)
    acc_value = tf.reduce_sum(loss_value)

    # mean_pred_policy = tf.reduce_mean(pred_policy)

    return pred_policy, pred_value, loss, optimizer, acc_policy, acc_value  # , mean_pred_policy
