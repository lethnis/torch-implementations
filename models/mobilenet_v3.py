import torch
from torch import nn

from ._options import MobileNetv3_options

# [kernel, expansion size, in_channels, out_channels, use SqueezeExcitation, activation function, stride]
options_dict: dict[MobileNetv3_options, list] = {
    "Large": [
        [3, 16, 16, 16, False, "relu", 1],
        [3, 64, 16, 24, False, "relu", 2],
        [3, 72, 24, 24, False, "relu", 1],
        [5, 72, 24, 40, True, "relu", 2],
        [5, 120, 40, 40, True, "relu", 1],
        [5, 120, 40, 40, True, "relu", 1],
        [3, 240, 40, 80, False, "hswish", 2],
        [3, 200, 80, 80, False, "hswish", 1],
        [3, 184, 80, 80, False, "hswish", 1],
        [3, 184, 80, 80, False, "hswish", 1],
        [3, 480, 80, 112, True, "hswish", 1],
        [3, 672, 112, 112, True, "hswish", 1],
        [5, 672, 112, 160, True, "hswish", 2],
        [5, 960, 160, 160, True, "hswish", 1],
        [5, 960, 160, 160, True, "hswish", 1],
    ],
    "Small": [
        [3, 16, 16, 16, True, "relu", 2],
        [3, 72, 16, 24, False, "relu", 2],
        [3, 88, 24, 24, False, "relu", 1],
        [5, 96, 24, 40, True, "hswish", 2],
        [5, 240, 40, 40, True, "hswish", 1],
        [5, 240, 40, 40, True, "hswish", 1],
        [5, 120, 40, 48, True, "hswish", 1],
        [5, 144, 48, 48, True, "hswish", 1],
        [5, 288, 48, 96, True, "hswish", 2],
        [5, 576, 96, 96, True, "hswish", 1],
        [5, 576, 96, 96, True, "hswish", 1],
    ],
}


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        act: nn.modules.activation = nn.ReLU(),
        groups: int = 1,
        bn: bool = True,
        bias: bool = False,
    ):
        """Basic convolutional block. Consists of conv -> batchnorm -> relu.
        Note: padding will be counted automatically.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (int): kernel size.
            stride (int): stride.
            act (nn.modules.activation, optional): activation function. Defaults to nn.ReLU().
            groups (int, optional): number of groups for separable conv. Defaults to 1.
            bn (bool, optional): use batchnorm or don't. Defaults fo True.
            bias (bool, optional): use bias or don't. Defaults fo False.
        """
        super().__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.act = act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    def __init__(self, in_channels: int):
        """Squeeze and Excitation block. Applies channel-wise attention to input channels.
        avgpool -> conv -> relu -> conv -> hard-sigmoid. Applies weights to each channels.

        Args:
            in_channels (int): number of input channels.
        """
        super().__init__()

        # reduced dim
        r = in_channels // 4
        # (batch, channels, height, width) -> (batch, channels, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # reduce number of channels
        self.conv1 = nn.Conv2d(in_channels, r, 1)
        # apply non-linearity
        self.relu = nn.ReLU()
        # restore number of channels
        self.conv2 = nn.Conv2d(r, in_channels, 1)
        # apply non-linearity
        self.hsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        return x * self.hsigmoid(self.conv2(self.relu(self.conv1(self.avgpool(x)))))


class InvertedResidualWithSE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        expansion_size: int,
        se: bool,
        act: str,
        stride: int,
    ):
        """Inverted residual block with linear bottleneck. Similar to MobileNetv2,
        but with Squeeze and Excitation.

        Note: residual connection will be used if input shape and output shape are the same.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (int): kernel size.
            expansion_size (int): increased number of channels in bottleneck.
            se (bool): use Squeeze and Excitation block or don't.
            act (str): what activation function to use. 'relu' or 'hswish'.
            stride (int): stride.
        """
        super().__init__()

        self.is_residual = in_channels == out_channels and stride == 1

        self.block = nn.Sequential(
            # point-wise conv to increase number of channels
            ConvBlock(in_channels, expansion_size, 1, 1, nn.ReLU() if act == "relu" else nn.Hardswish()),
            # depth-wise conv
            ConvBlock(
                expansion_size,
                expansion_size,
                kernel_size,
                stride,
                nn.ReLU() if act == "relu" else nn.Hardswish(),
                expansion_size,
            ),
            # use channels-wise attention
            SEBlock(expansion_size) if se is True else nn.Identity(),
            # point-wise conv to decrease number of channels
            ConvBlock(expansion_size, out_channels, 1, 1, act=nn.Identity()),
        )

    def forward(self, x):
        if self.is_residual:
            return x + self.block(x)
        return self.block(x)


class MobileNetv3(nn.Module):
    def __init__(
        self,
        model_name: MobileNetv3_options,
        in_channels: int = 3,
        num_classes: int = 1000,
        alpha: float = 1.0,
        dropout_rate: float = 0.8,
    ):
        """MobileNetv3 implementation as described in the original paper.
        Original paper: https://arxiv.org/abs/1905.02244.
        Paper walkthrough and implementation: https://www.youtube.com/watch?v=0oqs-inp7sA.

        Args:
            model_name (MobileNetv3_options): 'Small' or 'Large' model as in the paper.
            in_channels (int, optional): number of input channels (image channels). Defaults to 3.
            num_classes (int, optional): number of classes. Defaults to 1000.
            alpha (float, optional): multiplier for number of channels. Defaults to 1.0.
            dropout_rate (float, optional): dropout for classifier. Defaults to 0.8.
        """
        super().__init__()

        features = []
        # get the list of layers for provided model name
        options = options_dict[model_name]

        # add first layer
        features.append(ConvBlock(in_channels, int(16 * alpha), 3, 2, nn.Hardswish()))

        # add InvertedResidual blocks
        for block in options:
            kernel_size, expansion_size, in_channels, out_channels, se, act, stride = block
            in_channels = int(alpha * in_channels)
            out_channels = int(alpha * out_channels)
            expansion_size = int(alpha * expansion_size)
            features.append(
                InvertedResidualWithSE(in_channels, out_channels, kernel_size, expansion_size, se, act, stride)
            )

        self.features = nn.Sequential(*features)

        # get the proper final channels as in the paper
        last_output = 1280 if model_name == "Large" else 1024
        last_output = int(alpha * last_output)

        # final classifier
        self.classifier = nn.Sequential(
            ConvBlock(out_channels, expansion_size, 1, 1, nn.Hardswish()),
            nn.AdaptiveAvgPool2d(1),
            ConvBlock(expansion_size, last_output, 1, 1, nn.Hardswish(), bn=False, bias=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(last_output, num_classes, 1, 1),
        )

    def forward(self, x):
        return torch.flatten(self.classifier(self.features(x)), 1)
