import torch
from torch import nn


class DepthWiseSeparable(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """DepthWiseSeperable block of MobileNet which performs the following operations:
        (a) depthwise is 3x3 conv applied separately for each channel
        (b) pointwise is 1x1 conv

            Note:
                1. groups=in_channels used for depthwise conv
                2. in_channels and out_channels are same for depthwise conv
                3. bias = False due to the usage of BatchNorm

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            stride (int): used in depthwise conv to reduce feature maps sizes

        Attributes:
            depthwise separable conv block
        """

        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)

        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.bn1(self.depthwise(x)))

        return self.relu(self.bn2(self.pointwise(x)))


class MobileNetv1(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, alpha: float = 1.0):
        """Creates MobileNetv1 architecture repeating model from the original paper.
        Original paper: https://arxiv.org/pdf/1704.04861.pdf.
        Paper walkthrough: https://medium.com/@karuneshu21/mobilenets-v1-paper-walkthrough-92549d0fbbd.
        Implementation from paper: https://medium.com/@karuneshu21/implement-mobilenet-v1-in-pytorch-fd03a6618321.

        Args:
            in_channels (int, optional): number of image channels. Defaults to 3.
            num_classes (int, optional): number of classes for final classification layer. Defaults to 1000.
            alpha (float, optional): Multiplier for number of channels.
                Choose one of: 1.0, 0.75, 0.5, 0.25. Defaults to 1.0.

        Arguments:
            MobileNetv1 model
        """

        super().__init__()

        self.features = nn.Sequential(
            # initial conv layer
            nn.Conv2d(
                in_channels=in_channels, out_channels=int(32 * alpha), kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=int(32 * alpha)),
            # depthwise separable convs
            DepthWiseSeparable(in_channels=int(32 * alpha), out_channels=int(64 * alpha), stride=1),
            DepthWiseSeparable(in_channels=int(64 * alpha), out_channels=int(128 * alpha), stride=2),
            DepthWiseSeparable(in_channels=int(128 * alpha), out_channels=int(128 * alpha), stride=1),
            DepthWiseSeparable(in_channels=int(128 * alpha), out_channels=int(256 * alpha), stride=2),
            DepthWiseSeparable(in_channels=int(256 * alpha), out_channels=int(256 * alpha), stride=1),
            DepthWiseSeparable(in_channels=int(256 * alpha), out_channels=int(512 * alpha), stride=2),
            DepthWiseSeparable(in_channels=int(512 * alpha), out_channels=int(512 * alpha), stride=1),
            DepthWiseSeparable(in_channels=int(512 * alpha), out_channels=int(512 * alpha), stride=1),
            DepthWiseSeparable(in_channels=int(512 * alpha), out_channels=int(512 * alpha), stride=1),
            DepthWiseSeparable(in_channels=int(512 * alpha), out_channels=int(512 * alpha), stride=1),
            DepthWiseSeparable(in_channels=int(512 * alpha), out_channels=int(512 * alpha), stride=1),
            DepthWiseSeparable(in_channels=int(512 * alpha), out_channels=int(1024 * alpha), stride=2),
            DepthWiseSeparable(in_channels=int(1024 * alpha), out_channels=int(1024 * alpha), stride=1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(int(1024 * alpha), num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc1(x)
