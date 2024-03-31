import torch
from torch import nn
from torchview import draw_graph
from torchinfo import summary

from _options import Densenet_options

# settings for original paper models
DENSENET_OPTIONS: dict[Densenet_options, list] = {
    # list of: number of repetitions, growth factor, compression factor, is bottleneck
    "DenseNet121": [[6, 12, 24, 16], 32, 0.5, True],
    "DenseNet169": [[6, 12, 32, 32], 32, 0.5, True],
    "DenseNet201": [[6, 12, 48, 32], 32, 0.5, True],
    "DenseNet264": [[6, 12, 64, 48], 32, 0.5, True],
}


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_bottleneck: bool = False):
        """Basic layer of the dense net block.
        If is_bottleneck is True bn->relu->conv1x1 -> bn->relu->conv3x3.
        If is_bottleneck is False bn->relu->conv3x3

        Note:
            1. in bottleneck number of intermediate layers is 4 * out channels

        Args:
            in_channels (int): number of input channels (image channels)
            out_channels (int): number of output channels
            is_bottleneck (bool, optional): whether to use bottleneck conv1x1 or don't. Defaults to False.
        """

        super().__init__()

        self.is_bottleneck = is_bottleneck

        if self.is_bottleneck:

            self.bn1 = nn.BatchNorm2d(num_features=in_channels)
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=4 * out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )

            self.bn2 = nn.BatchNorm2d(num_features=4 * out_channels)
            self.conv2 = nn.Conv2d(
                in_channels=4 * out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )

        else:

            self.bn1 = nn.BatchNorm2d(num_features=in_channels)
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )

        self.relu = nn.ReLU()

    def forward(self, x):

        residual = x

        if self.is_bottleneck:
            x = self.conv1(self.relu(self.bn1(x)))
            x = self.conv2(self.relu(self.bn2(x)))

        else:
            x = self.conv1(self.relu(self.bn1(x)))

        x = torch.cat([residual, x], dim=1)

        return x


class DenseBlock(nn.Module):


def test_DenseLayer():
    x = torch.randn(1, 64, 224, 224)
    model = DenseLayer(64, 12, False)
    print(model(x).shape)
    draw_graph(model, x, save_graph=True, expand_nested=True)
    del x, model


test_DenseLayer()
