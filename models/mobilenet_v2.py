import torch
from torch import nn

blocks = [
    # expand ratio, out channels, num repeats, stride
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, groups: int = 1
    ):
        """Basic convolutional block. Consists of conv -> batchnorm -> relu6.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (int): kernel size.
            stride (int): stride.
            padding (int): padding.
            groups (int, optional): number of groups for separable conv. Defaults to 1.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int):
        """Inverted residual block with expansion as described in the original paper.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            stride (int): stride.
            expand_ratio (int): multiplier for intermediate number of channels.
        """

        super().__init__()

        # count number of channels for expansion
        expanded_channels = int(in_channels * expand_ratio)
        # do residual only if output shape is completely the same
        self.is_residual = in_channels == out_channels and stride == 1

        # if expand ratio is one (for first block in original paper) do depth-wise conv and point-wise conv
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # depth-wise conv (groups=in_channels)
                nn.Conv2d(in_channels, expanded_channels, 3, stride, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.ReLU6(True),
                # point-wise conv (kernel_size=1)
                nn.Conv2d(expanded_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv = nn.Sequential(
                # point-wise conv (kernel_size=1)
                nn.Conv2d(in_channels, expanded_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.ReLU6(True),
                # depth-wise conv (groups=in_channels)
                nn.Conv2d(expanded_channels, expanded_channels, 3, stride, 1, groups=expanded_channels, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.ReLU6(True),
                # point-wise conv (kernel_size=1)
                nn.Conv2d(expanded_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        if self.is_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetv2(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, alpha: float = 1.0):
        """MobileNetv2 implementation as described in the original paper.
        Original paper: https://arxiv.org/pdf/1801.04381.pdf.
        Explanations: https://youtu.be/eZzr780Qxfg?list=PLLCGSi_WZBNftPaTaX4k4AwLp4VreDAwV.
        Another implementations: https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py.

        Args:
            in_channels (int, optional): number of input channels (image channels). Defaults to 3.
            num_classes (int, optional): number of classes. Defaults to 1000.
            alpha (float, optional): multiplier for model width (number of channels). Defaults to 1.0.
        """
        super().__init__()

        # initialize first block
        features = [ConvBlock(in_channels, 32, 3, 2, 1)]
        in_channels = 32

        # add other blocks to the model. Stride only applies on first repetition.
        for expand_ratio, out_channels, num_repeats, stride in blocks:
            out_channels = int(alpha * out_channels)
            for i in range(num_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        stride if i == 0 else 1,
                        expand_ratio,
                    )
                )
                in_channels = out_channels

        # add last layer as described in the paper
        features.append(ConvBlock(in_channels, int(alpha * 1280), 1, 1, 0))

        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(int(alpha * 1280), num_classes)

    def forward(self, x):
        x = self.avgpool(self.features(x))
        return self.classifier(torch.flatten(x, 1))
