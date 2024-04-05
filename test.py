import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, act=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, kernel_size // 2, groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, r=24):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // r, 1)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(in_channels // r, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.conv2(self.silu(self.conv1(self.avgpool(x)))))


class MBBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion, r):
        super().__init__()
        exp_channels = in_channels * expansion
        self.add = in_channels == out_channels and stride == 1
        self.conv1 = ConvBlock(in_channels, exp_channels, 1, 1) if expansion == 1 else nn.Identity()
        self.conv2 = ConvBlock(exp_channels, exp_channels, kernel_size, stride, groups=exp_channels)
        self.se = SqueezeExcitation(exp_channels)
        self.conv3 = ConvBlock(exp_channels, out_channels, 1, 1, act=False)

    def forward(self, inputs):
        x = self.conv3(self.se(self.conv2(self.conv1(inputs))))
        if self.add:
            return inputs + x
        return x


class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_rate):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        return self.fc(self.dropout(self.pool(x)))
