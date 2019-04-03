import tensorflow as tf
from tensorflow import keras
from src.config import CFG


def ResidualBlock(input, training, regularizer):
    # Convolutional layer #1
    conv1 = keras.layers.Conv2D(
        filters=236,
        kernel_size=[3, 3],
        padding='same',
        strides=1,
        kernel_regularizer=regularizer
    )(input)
    # Batch normalization #1
    batchnorm1 = tf.layers.batch_normalization(conv1, training=training)
    # ReLU #1
    relu1 = keras.layers.Activation('relu')(batchnorm1)
    # Convolutional layer #2
    conv2 = keras.layers.Conv2D(
        filters=236,
        kernel_size=[3, 3],
        padding='same',
        strides=1,
        kernel_regularizer=regularizer
    )(relu1)
    # Batch normalization #2
    batchnorm2 = tf.layers.batch_normalization(conv2, training=training)
    # Skip connection
    skip = tf.add(batchnorm2, input)
    # ReLU #2
    relu2 = keras.layers.Activation('relu')(skip)
    return relu2


def ResidualTower(input, training, regularizer, n_blocks):
    res = input
    for _ in range(n_blocks):
        res = ResidualBlock(res, training, regularizer)
    return res


def AlphaGo19Net(inputs, training, pi, z):
    # Regularizer
    regularizer = keras.regularizers.l2(0.1)

    ### Body
    # Convolutional layer #1
    conv1 = keras.layers.Conv2D(
        filters=236,
        kernel_size=[3, 3],
        padding='same',
        strides=1,
        kernel_regularizer=regularizer
    )(inputs)
    # Batch normalization layer #1
    batchnorm1 = tf.layers.batch_normalization(conv1, training=training)
    # ReLU layer #1
    relu2 = keras.layers.Activation('relu')(batchnorm1)
    # Tower of residual blocks
    tower = ResidualTower(relu2, training, regularizer, CFG.resnet_blocks)

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
    batchnorm4 = tf.layers.batch_normalization(conv3, training=training)
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
    batchnorm5 = tf.layers.batch_normalization(conv4, training=training)
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
        # loss_value = tf.reshape(tf.squared_difference(z, pred_value), (-1,))
        loss_value = tf.losses.mean_squared_error(z, pred_value)
        # loss_policy = tf.reduce_mean(tf.multiply(pi, tf.math.log(1e-6 + pred_policy)), axis=1)
        # loss_policy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=pi, logits=fc1))
        loss_policy = tf.losses.softmax_cross_entropy(pi, pred_policy)
        # loss_policy = tf.reduce_sum(tf.multiply(pi, tf.math.log(pred_policy)), axis=1)
        regularization = CFG.l2_val * tf.reduce_sum(regularization_losses)
        # loss = 0.1 * loss_value + 0.9 * loss_policy + regularization
        loss = loss_value + loss_policy + regularization

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
