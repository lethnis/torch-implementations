import torch
from torch import nn

from ._options import Densenet_options

# settings for original paper models
DENSENET_OPTIONS: dict[Densenet_options, list] = {
    # list of: number of repetitions, growth factor, compression factor, is bottleneck
    "DenseNet121": [[6, 12, 24, 16], 32, 0.5, True],
    "DenseNet169": [[6, 12, 32, 32], 32, 0.5, True],
    "DenseNet201": [[6, 12, 48, 32], 32, 0.5, True],
    "DenseNet264": [[6, 12, 64, 48], 32, 0.5, True],
}


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, k: int, is_bottleneck: bool = False):
        """Basic layer of the dense net block.
        If is_bottleneck is True bn->relu->conv1x1 -> bn->relu->conv3x3.
        If is_bottleneck is False bn->relu->conv3x3

        Note:
            1. in bottleneck number of intermediate layers is 4 * k
            2. output of the the DenseLayer is k

        Args:
            in_channels (int): number of input channels (image channels)
            k (int): growth rate for bottleneck expansion in channels, also output channels
            is_bottleneck (bool, optional): whether to use bottleneck conv1x1 or don't. Defaults to False.
        """

        super().__init__()

        self.is_bottleneck = is_bottleneck

        if self.is_bottleneck:

            self.bn1 = nn.BatchNorm2d(num_features=in_channels)
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=4 * k,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )

            self.bn2 = nn.BatchNorm2d(num_features=4 * k)
            self.conv2 = nn.Conv2d(
                in_channels=4 * k,
                out_channels=k,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )

        else:

            self.bn1 = nn.BatchNorm2d(num_features=in_channels)
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=k,
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
    def __init__(self, layer_rep: int, in_channels: int, k: int = 32, is_bottleneck: bool = False):
        """Basic block of a DenseNet. Generates DenseLayers and concatenates theirs outputs.

        Note:
            1. if is_bottleneck == True DenseLayer will have bn->relu->conv1x1 -> bn->relu->conv3x3.
            2. if is_bottleneck == False DenseLayer will have bn->relu->conv3x3.
            3. output of conv1x1 will be k*4
            4. output of conv3x3 and hence DenseLayer will be k

        Args:
            layer_rep (int): how many time to repeat DenseLayer
            in_channels (int): number of input channels (image channels for the very first block)
            k (int, optional): growth rate. How many channels will return every DenseLayer. Defaults to 32.
            is_bottleneck (bool, optional): Whether to use bottleneck structure or don't. Defaults to False.

        Attributes:
            Sequence of DenseLayers with skip connections
        """

        super().__init__()

        self.deep_nn = nn.ModuleList()

        for i in range(layer_rep):
            self.deep_nn.add_module(
                f"DenseLayer_{i}",
                DenseLayer(in_channels=in_channels + k * i, k=k, is_bottleneck=is_bottleneck),
            )

    def forward(self, x):
        for layer in self.deep_nn:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels: int, compression_factor: int = 0.5):
        """Layer reduces number of input channels in conv1x1 layer.
        Then halves feature map size (img size) in average pooling layer.

        Args:
            in_channels (int): number of input channels
            compression_factor (int, optional): multiplier to reduce the number of input channels. Defaults to 0.5.
        """

        super().__init__()

        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * compression_factor),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.avgpool(self.conv(self.bn(x)))


class DenseNet(nn.Module):
    def __init__(
        self,
        repetitions: list,
        k: int = 32,
        compression_factor: float = 0.5,
        is_bottleneck: bool = False,
        in_channels: int = 3,
        num_classes: int = 1000,
        start_channels: int = 64,
    ):
        """DenseNet implementation according to the original paper.
        Original paper: https://arxiv.org/pdf/1608.06993.pdf
        Paper walkthrough: https://medium.com/@karuneshu21/densenet-paper-walkthrough-764481fced3a
        Implementation from paper: https://medium.com/@karuneshu21/implement-densenet-in-pytorch-46374ef91900

        Input goes through conv7x7 and maxpool2x2 and then DenseBlocks.

        Args:
            repetitions (list): how many times to repeat DenseLayer in every DenseBlock
            k (int, optional): growth rate. Every DenseLayer layer return this number of channels to stack in DenseBlock. Defaults to 32.
            compression_factor (float, optional): multiplier for TransitionLayer between DenseBlocks. Reduces amount of channels. Defaults to 0.5.
            is_bottleneck (bool, optional): whether to use bottleneck structure or don't. Defaults to False.
            in_channels (int, optional): number of input channels (image channels). Defaults to 3.
            num_classes (int, optional): total number of classes for multiplication. Defaults to 1000.
            start_channels (int, optional): channels after first conv and maxpool. Defaults to 64.
        """

        super().__init__()

        # at start we use conv7x7 and maxpool2x2 according to the paper
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=start_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_features=start_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # start iterating through blocks and transitions except for final time
        self.deep_nn = nn.ModuleList()
        block_in_channels = start_channels

        for i in range(len(repetitions) - 1):
            self.deep_nn.add_module(
                f"DenseBlock_{i+1}",
                DenseBlock(layer_rep=repetitions[i], in_channels=block_in_channels, k=k, is_bottleneck=is_bottleneck),
            )
            block_in_channels = int(block_in_channels + k * repetitions[i])

            self.deep_nn.add_module(
                f"TransitionLayer_{i+1}",
                TransitionLayer(in_channels=block_in_channels, compression_factor=compression_factor),
            )
            block_in_channels = int(block_in_channels * compression_factor)

        # final DenseBlock without transition
        self.deep_nn.add_module(
            f"DenseBlock_{i+2}",
            DenseBlock(layer_rep=repetitions[-1], in_channels=block_in_channels, k=k, is_bottleneck=is_bottleneck),
        )
        block_in_channels = int(block_in_channels + k * repetitions[-1])

        self.bn2 = nn.BatchNorm2d(num_features=block_in_channels)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # for binary classification we need only 1 output
        if num_classes == 2:
            num_classes = 1

        self.fc = nn.Linear(in_features=block_in_channels, out_features=num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv(x))))

        for layer in self.deep_nn:
            x = layer(x)

        x = self.avgpool(self.relu(self.bn2(x)))

        x = torch.flatten(x, start_dim=1)

        return self.fc(x)

    @classmethod
    def from_options(
        cls, options: Densenet_options, in_channels: int = 3, num_classes: int = 1000, start_channels: int = 64
    ):
        """Creates an instance of ResNet from the original paper.
        Options are: repetitions in every DenseBlock, growth factor, compression factor, is bottleneck.

        Args:
            options (Densenet_options): options of the model. One of (DenseNet121/169/201/264)
            in_channels (int, optional): number of input channels (image channels). Defaults to 3.
            num_classes (int, optional): total number of classes for classification. Defaults to 1000.
            start_channels (int, optional): channels after first conv and maxpool, before DenseBlocks. Defaults to 64.

        Returns:
            DenseNet: in instance of chosen DenseNet model
        """
        return DenseNet(
            *DENSENET_OPTIONS[options], in_channels=in_channels, num_classes=num_classes, start_channels=start_channels
        )
