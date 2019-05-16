import torch
import torch.nn as nn
from config import CFG


def conv3x3(in_channels, out_channels, stride=1, kernel_size=3, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
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
    def __init__(self, block):
        super(ResNet, self).__init__()
        self.resblock = block(CFG.num_filters, CFG.num_filters)

    def forward(self, x):
        out = x
        for _ in range(CFG.resnet_blocks):
            out = self.resblock(x)
        return out


class Body(nn.Module):
    def __init__(self, in_channels):
        super(Body, self).__init__()
        self.conv1 = conv3x3(in_channels, CFG.num_filters)
        self.bn1 = nn.BatchNorm2d(CFG.num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.tower = ResNet(ResidualBlock)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.tower(out)
        return out


class PolicyHead(nn.Module):
    def __init__(self, num_classes):
        super(PolicyHead, self).__init__()
        self.conv1 = conv3x3(
            CFG.num_filters, out_channels=2, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(2 * 6 * 7, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.softmax(out)
        return out


class ValueHead(nn.Module):
    def __init__(self):
        super(ValueHead, self).__init__()
        self.conv1 = conv3x3(
            CFG.num_filters, out_channels=1, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(1 * 6 * 7, CFG.num_filters)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(CFG.num_filters, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.relu2(out)
        out = self.linear2(out)
        out = self.tanh(out)
        return out


class AlphaGoNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AlphaGoNet, self).__init__()
        self.body = Body(in_channels)
        self.policy_head = PolicyHead(num_classes)
        self.value_head = ValueHead()

    def forward(self, x):
        out = self.body(x)
        policy = self.policy_head(out)
        value = self.value_head(out)
        return policy, value


class AlphaLoss(nn.Module):
    """
    Custom loss as defined in the paper :
    (z - v) ** 2 --> MSE Loss
    (-pi * logp) --> Cross Entropy Loss
    z : self_play_winner
    v : winner
    pi : self_play_probas
    p : probas

    The loss is then averaged over the entire batch
    """

    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, winner, self_play_winner, probas, self_play_probas):
        value_error = (self_play_winner.view(-1) - winner) ** 2
        policy_error = torch.sum((-self_play_probas * (1e-6 + probas).log()), 1)
        total_error = (value_error.view(-1) + policy_error).mean()
        return total_error, value_error, policy_error
