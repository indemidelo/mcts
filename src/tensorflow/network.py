import tensorflow as tf
from tensorflow import keras
from config import CFG


def ResidualBlock(input, is_train):
    # Convolutional layer #1
    conv1 = keras.layers.Conv2D(
        filters=CFG.num_filters,
        kernel_size=[3, 3],
        padding='same',
        strides=1,
        data_format='channels_first'
    )(input)
    # Batch normalization #1
    batchnorm1 = keras.layers.BatchNormalization(axis=1)(conv1, training=is_train)
    # ReLU #1
    relu1 = keras.layers.Activation('relu')(batchnorm1)
    # Convolutional layer #2
    conv2 = keras.layers.Conv2D(
        filters=CFG.num_filters,
        kernel_size=[3, 3],
        padding='same',
        strides=1,
        data_format='channels_first'
    )(relu1)
    # Batch normalization #2
    batchnorm2 = keras.layers.BatchNormalization(axis=1)(conv2, training=is_train)
    # Skip connection
    skip = tf.add(batchnorm2, input)
    # ReLU #2
    relu2 = keras.layers.Activation('relu')(skip)
    return relu2


def ResidualTower(input, n_blocks, is_train):
    res = input
    for _ in range(n_blocks):
        res = ResidualBlock(res, is_train)
    return res


def AlphaGo19Net(inputs, outputs, pi, z, is_train):

    ### Body
    # Convolutional layer #1
    conv1 = keras.layers.Conv2D(
        filters=CFG.num_filters,
        kernel_size=[3, 3],
        padding='same',
        strides=1,
        data_format='channels_first'
    )(inputs)
    # Batch normalization layer #1
    batchnorm1 = keras.layers.BatchNormalization(axis=1)(conv1, training=is_train)
    # ReLU layer #1
    relu2 = keras.layers.Activation('relu')(batchnorm1)
    # Tower of residual blocks
    tower = ResidualTower(relu2, CFG.resnet_blocks, is_train)

    ### Policy head
    # Convolutional layer #3
    conv3 = keras.layers.Conv2D(
        filters=2,
        kernel_size=[1, 1],
        padding='same',
        strides=1,
        data_format='channels_first'
    )(tower)
    # Batch normalization layer #4
    batchnorm4 = keras.layers.BatchNormalization(axis=1)(conv3, training=is_train)
    # ReLU layer #4
    relu4 = keras.layers.Activation('relu')(batchnorm4)
    # Fully connected layer #1
    with tf.name_scope('PolicyHead'):
        relu4 = keras.layers.Flatten()(relu4)
        fc1 = keras.layers.Dense(units=outputs)(relu4)
        pred_policy = keras.layers.Activation('softmax')(fc1)

    ### Value Head
    # Convolutional layer #4
    conv4 = keras.layers.Conv2D(
        filters=1,
        kernel_size=[1, 1],
        strides=1,
        data_format='channels_first'
    )(tower)
    # Batch normalization #5
    batchnorm5 = keras.layers.BatchNormalization(axis=1)(conv4, training=is_train)
    # ReLU #5
    relu5 = keras.layers.Activation('relu')(batchnorm5)
    # Fully connected layer #2
    relu5 = keras.layers.Flatten()(relu5)
    fc2 = keras.layers.Dense(units=CFG.num_filters)(relu5)
    # ReLU #6
    relu6 = keras.layers.Activation('relu')(fc2)
    # Fully connected layer #3
    fc_output = keras.layers.Dense(units=1)(relu6)
    # Tanh activator
    pred_value = keras.layers.Activation('tanh')(fc_output)
    # Reshape predicted value
    pred_value = tf.reshape(pred_value, shape=[-1, ])

    ### Loss
    with tf.name_scope('Loss'):
        loss_value = tf.losses.mean_squared_error(z, pred_value)
        loss_policy = tf.losses.softmax_cross_entropy(pi, pred_policy)
        # loss = 0.1 * loss_value + 0.9 * loss_policy
        loss = loss_value + loss_policy

    # Configure optimizer
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=CFG.learning_rate,
        momentum=CFG.momentum).minimize(loss)

    # Accuracy
    # with tf.name_scope('Accuracy'):
    # todo fix this accuracy
    # acc_policy = tf.reduce_mean(pred_policy - pi)
    # acc_value = tf.reduce_mean(pred_value - z)

    return pred_policy, pred_value, loss, optimizer, loss_policy, loss_value
