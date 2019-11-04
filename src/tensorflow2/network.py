import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Layer, BatchNormalization, ReLU
from tensorflow.keras import Model, Sequential
from config import CFG


def conv3x3(out_channels):
    return Conv2D(filters=out_channels, kernel_size=3,
                  padding='same', data_format='channels_first')


class ResidualBlock(Layer):
    def __init__(self, out_channels, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(out_channels)
        self.bn1 = BatchNormalization()
        self.relu = ReLU()
        self.conv2 = conv3x3(out_channels)
        self.bn2 = BatchNormalization()
        self.downsample = downsample

    def call(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResidualTower(Layer):
    def __init__(self):
        super(ResidualTower, self).__init__()
        self.tower = self.make_tower()

    def make_tower(self):
        layers = list()
        for _ in range(CFG.resnet_blocks):
            layers.append(ResidualBlock(CFG.num_filters))
        return Sequential(layers)

    def call(self, x):
        out = self.tower(x)
        return out


class Body(Layer):
    def __init__(self):
        super(Body, self).__init__()
        self.conv = conv3x3(CFG.num_filters)
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.tower = ResidualTower()

    def call(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.tower(out)
        return out


class PolicyHead(Layer):
    def __init__(self, num_classes):
        super(PolicyHead, self).__init__()
        self.conv1 = Conv2D(
            filters=2, kernel_size=1, padding='valid')
        self.bn = BatchNormalization()
        self.flat = Flatten()
        self.relu = ReLU()
        self.linear = Dense(num_classes, activation='softmax')

    def call(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.flat(out)
        out = self.relu(out)
        # out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ValueHead(Layer):
    def __init__(self):
        super(ValueHead, self).__init__()
        self.conv = Conv2D(
            filters=1, kernel_size=1, padding='valid')
        self.bn = BatchNormalization()
        self.flat = Flatten()
        self.relu = ReLU()
        self.linear1 = Dense(CFG.num_filters, activation='relu')
        self.linear2 = Dense(1, activation='tanh')

    def call(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.flat(out)
        out = self.relu(out)
        # out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out


class AlphaGoNet(Model):
    def __init__(self, game):
        super(AlphaGoNet, self).__init__()
        policy_len = game.policy_shape()
        self.body = Body()
        self.policy_head = PolicyHead(policy_len)
        self.value_head = ValueHead()

    def call(self, inputs):
        out = self.body(inputs)
        policy = self.policy_head(out)
        value = self.value_head(out)
        return policy, value


@tf.function()
def alpha_loss(y_true, y_pred):
    z, pi = y_true
    pred_value, pred_policy = y_pred
    loss_value = tf.losses.mean_squared_error(z, pred_value)
    loss_policy = tf.losses.softmax_cross_entropy(pi, pred_policy)
    # loss = 0.1 * loss_value + 0.9 * loss_policy
    loss = loss_value + loss_policy
    return loss, loss_policy, loss_value

# def alpha_loss(z, pred_value, pi, pred_policy):
#     loss_value = tf.losses.mean_squared_error(z, pred_value)
#     loss_policy = tf.losses.softmax_cross_entropy(pi, pred_policy)
#     # loss = 0.1 * loss_value + 0.9 * loss_policy
#     loss = loss_value + loss_policy
#     return loss, loss_policy, loss_value
