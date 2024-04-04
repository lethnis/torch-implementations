import torch
from torch import nn

from ._options import Resnet_options


# setting for original models
RESNET_OPTIONS: dict[Resnet_options, list] = {
    # list of: number of channels, number of repetitions, expansion factor, is Bottleneck
    "ResNet18": [[64, 128, 256, 512], [2, 2, 2, 2], 1, False],
    "ResNet34": [[64, 128, 256, 512], [3, 4, 6, 3], 1, False],
    "ResNet50": [[64, 128, 256, 512], [3, 4, 6, 3], 4, True],
    "ResNet101": [[64, 128, 256, 512], [3, 4, 23, 3], 4, True],
    "ResNet152": [[64, 128, 256, 512], [3, 8, 36, 3], 4, True],
}


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, intermediate_channels: int, expansion: int, is_bottleneck: bool, stride: int):
        """Creates a block of a ResNet with residual connections. It may be basic or bottleneck.
        Basic block consists of conv 3x3 -> conv 3x3 layers.
        Bottleneck block consists of conv 1x1 -> conv 3x3 -> conv 1x1 layers.

        Note:
            1. Addition of feature maps occurs just before the final ReLU.
            2. If input size is different from output uses projected mapping.
            3. Bottleneck blocks are required for ResNet 50+.

        Args:
            in_channels (int): number of input channels.
            intermediate_channels (int): number of channels to 3x3 conv.
            expansion (int): factor by which the number of intermediate channels are multiplied.
            is_bottleneck (bool): whether to use Bottleneck block or don't.
            stride (int): stride applied in the 3x3 conv.

        Attributes:
            ResNet block with residual connections
        """
        super().__init__()

        self.is_bottleneck = is_bottleneck

        # if input channels == output channels (intermediate channels * expansion) use identity function
        if in_channels == intermediate_channels * expansion:
            self.identity = True
        else:
            self.identity = False
            self.projection = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=intermediate_channels * expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=intermediate_channels * expansion),
            )

        # commonly used relu
        self.relu = nn.ReLU()

        # is_bottleneck == True for ResNet 50+
        if self.is_bottleneck:

            # conv 1x1 for reducing channels
            self.conv1_1x1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=intermediate_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.batchnorm1 = nn.BatchNorm2d(num_features=intermediate_channels)

            # conv 3x3 for extracting features and optional feature maps reducing (if stride=2)
            self.conv2_3x3 = nn.Conv2d(
                in_channels=intermediate_channels,
                out_channels=intermediate_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.batchnorm2 = nn.BatchNorm2d(num_features=intermediate_channels)

            # conv 1x1 for increasing number of channels
            self.conv3_1x1 = nn.Conv2d(
                in_channels=intermediate_channels,
                out_channels=intermediate_channels * expansion,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.batchnorm3 = nn.BatchNorm2d(num_features=intermediate_channels * expansion)

        else:
            # conv 3x3 optional feature maps reducing (if stride=2)
            self.conv1_3x3 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=intermediate_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.batchnorm1 = nn.BatchNorm2d(num_features=intermediate_channels)

            # conv 3x3 for extracting features, no dimensions changed
            self.conv2_3x3 = nn.Conv2d(
                in_channels=intermediate_channels,
                out_channels=intermediate_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.batchnorm2 = nn.BatchNorm2d(num_features=intermediate_channels)

    def forward(self, x):
        # store input for addition before the final relu
        residual = x

        if self.is_bottleneck:
            x = self.relu(self.batchnorm1(self.conv1_1x1(x)))
            x = self.relu(self.batchnorm2(self.conv2_3x3(x)))
            x = self.batchnorm3(self.conv3_1x1(x))

        else:
            x = self.relu(self.batchnorm1(self.conv1_3x3(x)))
            x = self.batchnorm2(self.conv2_3x3(x))

        # if number of channels remains the same simply use addition
        if self.identity:
            x += residual
        else:
            x += self.projection(residual)

        # final relu
        return self.relu(x)


class ResNet(nn.Module):
    def __init__(
        self,
        channels_list: list,
        repetition_list: list,
        expansion: int,
        is_bottleneck: bool,
        in_channels: int,
        num_classes: int,
    ):
        """Creates the ResNet architecure based on the provided options.
        Original paper: https://arxiv.org/pdf/1512.03385.pdf
        Paper walkthrough: https://medium.com/@karuneshu21/resnet-paper-walkthrough-b7f3bdba55f0
        Implementation from paper: https://medium.com/@karuneshu21/how-to-resnet-in-pytorch-9acb01f36cf5

        Args:
            channels_list (list): list with number of channels for every block
            repetition_list (list): list of number of repetitions for every block
            expansion (int): factor by which channels are multiplied
            is_bottleneck (bool): whether to use Bottleneck block or don't.
            in_channels (int): image channels
            num_classes (int): output number of classes
        """
        super().__init__()

        assert len(channels_list) == len(repetition_list)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.batchnorm1 = nn.BatchNorm2d(channels_list[0])
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        blocks = []

        blocks.append(
            self._make_block(
                in_channels=channels_list[0],
                intermediate_channels=channels_list[0],
                num_repeat=repetition_list[0],
                expansion=expansion,
                is_bottleneck=is_bottleneck,
                stride=1,
            )
        )

        for i in range(len(channels_list) - 1):
            blocks.append(
                self._make_block(
                    in_channels=channels_list[i] * expansion,
                    intermediate_channels=channels_list[i + 1],
                    num_repeat=repetition_list[i + 1],
                    expansion=expansion,
                    is_bottleneck=is_bottleneck,
                    stride=2,
                )
            )

        self.blocks = nn.Sequential(*blocks)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # for binary classification we need only 1 output
        if num_classes == 2:
            num_classes = 1

        self.fc1 = nn.Linear(in_features=channels_list[-1] * expansion, out_features=num_classes)

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv1(x)))

        x = self.avgpool(self.blocks(self.maxpool(x)))

        x = torch.flatten(input=x, start_dim=1)

        return self.fc1(x)

    def _make_block(
        self,
        in_channels: int,
        intermediate_channels: int,
        num_repeat: int,
        expansion: int,
        is_bottleneck: bool,
        stride: int,
    ):
        """Creates a block of a ResNet with residual connections.

        Args:
            in_channels (int): number of input channels.
            intermediate_channels (int): number of channels to 3x3 conv.
            num_repeat (int): number of repetitions of the blocks.
            expansion (int): factor by which the number of intermediate channels are multiplied.
            is_bottleneck (bool): whether to use Bottleneck block or don't.
            stride (int): stride applied in the 3x3 conv. Only for the first block.

        Returns:
            nn.Sequential: Sequence of ResNet blocks.
        """

        layers = []

        layers.append(
            ResNetBlock(
                in_channels=in_channels,
                intermediate_channels=intermediate_channels,
                expansion=expansion,
                is_bottleneck=is_bottleneck,
                stride=stride,
            )
        )

        for _ in range(num_repeat - 1):
            layers.append(
                ResNetBlock(
                    in_channels=intermediate_channels * expansion,
                    intermediate_channels=intermediate_channels,
                    expansion=expansion,
                    is_bottleneck=is_bottleneck,
                    stride=1,
                )
            )

        return nn.Sequential(*layers)

    @classmethod
    def from_options(cls, options: Resnet_options, in_channels: int = 3, num_classes: int = 1000):
        """Creates an instance of ResNet from original paper.
        Options are: channels list, repetitions list, expansion factor, is bottleneck

        Args:
            options (Resnet_options): options of the model (ResNet18/34/50/101/152)
            in_channels (int): image channels
            num_classes (int): output number of classes

        Returns:
            ResNet: an instance of chosen ResNet model
        """
        return ResNet(*RESNET_OPTIONS[options], in_channels=in_channels, num_classes=num_classes)
