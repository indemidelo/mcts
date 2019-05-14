import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from config import CFG


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
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


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer = self._make_layer(block, 64, num_blocks, stride=1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer(out)
        # out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class Body(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, stride=1, downsample=None):
        super(Body, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.resnet = ResNet(ResidualBlock, CFG.resnet_blocks, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.resnet(out)
        return out

class PolicyHead(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(PolicyHead, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(out_channels, num_classes)
        self.softmax = F.softmax(num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.softmax(out)
        return out


class ValueHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ValueHead, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(out_channels, CFG.num_filters)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(CFG.num_filters, 1)
        self.tanh = F.tanh(1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.relu2(out)
        out = self.tanh(out)
        return out


def AlphaGo19Net(inputs, outputs, pi, z):
    ### Body
    # Convolutional layer #1
    conv1 = conv3x3(inputs, CFG.num_filters)
    conv1 = keras.layers.Conv2D(
        filters=CFG.num_filters,
        kernel_size=[3, 3],
        padding='same',
        strides=1,
        data_format='channels_first'
    )(inputs)
    # Batch normalization layer #1
    batchnorm1 = keras.layers.BatchNormalization()(conv1)
    # ReLU layer #1
    relu2 = keras.layers.Activation('relu')(batchnorm1)
    # Tower of residual blocks
    tower = ResidualTower(relu2, CFG.resnet_blocks)

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
    batchnorm4 = keras.layers.BatchNormalization()(conv3)
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
    batchnorm5 = keras.layers.BatchNormalization()(conv4)
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
