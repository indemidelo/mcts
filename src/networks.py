import torch.nn as nn


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.conv2d = nn.Conv2d(input_size, hidden_size,
                                kernel_size=3, stride=1, padding=1)
        self.conv2d_bn = nn.BatchNorm2d(hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.res = ResidualBlock(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.conv2d(x)
        out = self.conv2d_bn(out)
        out = self.relu(out)
        out = self.res(out)
        out = self.fc2(out)
        return out

def conv3x3(in_channels, out_channels, stride=1):
    """
    3x3 convolution
    :param in_channels:
    :param out_channels:
    :param stride:
    :return:
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# Residual block
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
