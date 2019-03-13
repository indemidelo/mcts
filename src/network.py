import tensorflow as tf


def ResidualBlock(input, regularizer):
    # Convolutional layer #1
    conv1 = tf.layers.conv2d(
        inputs=input,
        filters=236,
        kernel_size=[4, 4],
        padding='same',
        strides=1,
        kernel_regularizer=regularizer
    )

    # Batch normalization #1
    batchnorm1 = tf.layers.batch_normalization(conv1)

    # ReLU #1
    relu1 = tf.nn.relu(batchnorm1)

    # Convolutional layer #2
    conv2 = tf.layers.conv2d(
        inputs=relu1,
        filters=236,
        kernel_size=[4, 4],
        padding='same',
        strides=1,
        kernel_regularizer=regularizer
    )

    # Batch normalization #2
    batchnorm2 = tf.layers.batch_normalization(conv2)

    # Skip connection
    skip = tf.add(batchnorm2, input)

    # ReLU #2
    relu2 = tf.nn.relu(skip)

    return relu2


def ResidualTower(input, regularizer, n_blocks):
    res = input
    for _ in range(n_blocks):
        res = ResidualBlock(res, regularizer)
    return res


def AlphaGo19Net(inputs, pi, z, beta, n_res_blocks, learning_rate):
    # Regularizer
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    # Convolutional layer #1
    conv1 = tf.layers.conv2d(
        inputs=inputs,
        filters=236,
        kernel_size=[4, 4],
        padding='same',
        strides=1,
        kernel_regularizer=regularizer
    )

    # Batch normalization layer #1
    batchnorm1 = tf.layers.batch_normalization(conv1)

    # ReLU layer #1
    relu2 = tf.nn.relu(batchnorm1)

    # Tower of residual blocks
    tower = ResidualTower(relu2, regularizer, n_res_blocks)

    ### Policy head

    # Convolutional layer #3
    conv3 = tf.layers.conv2d(
        inputs=tower,
        filters=2,
        kernel_size=[1, 1],
        padding='same',
        strides=1,
        kernel_regularizer=regularizer
    )

    # Batch normalization layer #4
    batchnorm4 = tf.layers.batch_normalization(conv3)

    # ReLU layer #4
    relu4 = tf.nn.relu(batchnorm4)

    # Fully connected layer
    with tf.name_scope('Policy Head'):
        relu4 = tf.reshape(relu4, [-1, 6 * 7 * 3])
        pred_policy = tf.layers.dense(
            inputs=relu4,
            units=7,
            kernel_regularizer=regularizer
        )

    ### Value Head
    conv4 = tf.layers.conv2d(
        inputs=tower,
        filters=1,
        kernel_size=[1, 1],
        strides=1,
        kernel_regularizer=regularizer
    )
    batchnorm5 = tf.layers.batch_normalization(conv4)
    relu5 = tf.nn.relu(batchnorm5)
    relu5_reshaped = tf.reshape(relu5, [-1, 6 * 7 * 3])
    fc2 = tf.layers.dense(
        inputs=relu5_reshaped,
        units=256,
        kernel_regularizer=regularizer
    )
    relu6 = tf.nn.relu(fc2)
    fc_output = tf.layers.dense(
        inputs=relu6,
        units=1,
        kernel_regularizer=regularizer
    )
    pred_value = tf.nn.tanh(fc_output)

    # Collect all regularization losses
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    # Loss
    with tf.name_scope('Loss'):
        loss_value = tf.losses.mean_squared_error(labels=z, predictions=pred_value)
        loss_policy = tf.matmul(pi, tf.log(pred_policy), transpose_b=True)
        regularization = beta * tf.reduce_sum(regularization_losses)
        loss = loss_value + loss_policy + regularization

    # Configure optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # Accuracy
    # with tf.name_scope('Accuracy'):
    acc_policy = tf.reduce_mean(pred_policy - pi)
    acc_value = tf.reduce_mean(pred_value - z)

    # train_op = optimizer.minimize(
    #     loss=loss,
    #     global_step=tf.train.global_step()
    # )

    return pred_policy, pred_value, loss, optimizer, acc_policy, acc_value
