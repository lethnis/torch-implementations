import torch
from torch import nn


class ConvBlock(nn.Module):

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, bias: bool = False
    ):
        """Creates a conv layer followed by batchnorm and relu. Bias is False as batchnorm will nullify it anyways.

        Args:
            in_channels (int): number of input channels of the conv layer.
            out_channels (int): number of output channels of the conv layer.
            kernel_size (int): filter size.
            stride (int): number of pixels that the conv filter moves.
            padding (int): extra zero pixels around the border which affect the size of output feature map.
            bias (bool, optional): whether to use bias on conv layer or don't. Defaults to False.

        Attributes:
            creates conv, batchnorm and relu layers.
        """

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        self.bn = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_1x1: int,
        red_3x3: int,
        out_3x3: int,
        red_5x5: int,
        out_5x5: int,
        out_1x1_pooling: int,
    ):
        """Building block of Inception-v1 architecture. Creates following 4 branches and concatenates them.
        (a) branch1: 1x1 conv
        (b) branch2: 1x1 conv followed by 3x3 conv
        (c) branch3: 1x1 conv followed by 5x5 conv
        (d) branch4: Maxpool2d followed by 1x1 conv

        Note:
            1. output and input feature map height and width should remain the same. Only the channel output should change. e.g. 28x28x192 -> 28x28x256
            2. To generate same height and width of output feature map as the input feature map, following should be padding for
                * 1x1 conv: padding=0
                * 3x3 conv: padding=1
                * 5x5 conv: padding=2

        Args:
            in_channels (int): number of input channels
            out_1x1 (int): number of output channels for branch 1
            red_3x3 (int): reduced 3x3 referring to output channels of 1x1 conv just before 3x3 in branch2
            out_3x3 (int): number of output channels for branch 2
            red_5x5 (int): reduced 5x5 referring to output channels of 1x1 conv just before 5x5 in branch3
            out_5x5 (int): number of output channels for branch 3
            out_1x1_pooling (int): number of output channels for branch 4

        Attributes:
            concatenated feature maps from all 4 branches constituting output of Inception module.
        """
        super().__init__()

        # branch 1: conv 1x1
        self.branch1 = ConvBlock(in_channels, out_1x1, 1, 1, 0)

        # branch 2: reducing conv 1x1 -> conv 3x3
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, 1, 1, 0),
            ConvBlock(red_3x3, out_3x3, 3, 1, 1),
        )

        # branch 3: reducing conv 1x1 -> conv 5x5
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, 1, 1, 0),
            ConvBlock(red_5x5, out_5x5, 5, 1, 2),
        )

        # branch 4: max_pool 3x3 -> conv 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_1x1_pooling, 1, 1, 0),
        )

    def forward(self, x):

        # concatenation from dim=1 as dim=0 represents batch dimension
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class Inceptionv1(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        """Step-by-step building of the inceptionv1 architecture.
        Original paper: https://arxiv.org/pdf/1409.4842.pdf.
        Paper walkthrough: https://medium.com/@karuneshu21/inception-v1-paper-walkthrough-780a6910db46.
        Implementation from paper: https://medium.com/@karuneshu21/implement-inception-v1-in-pytorch-66bdbb3d0005.

        Args:
            in_channels (int): input channels. 3 for RGB image
            num_classes (int): number of classes in dataset

        Attributes:
            inceptionv1 model
        """
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64, 7, 2, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Sequential(ConvBlock(64, 64, 1, 1, 0), ConvBlock(64, 192, 3, 1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)

        # for binary classification we need only 1 output
        if num_classes == 2:
            num_classes = 1

        self.fc1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))

        x = self.maxpool3(self.inception3b(self.inception3a(x)))

        x = self.maxpool4(self.inception4e(self.inception4d(self.inception4c(self.inception4b(self.inception4a(x))))))

        x = self.dropout(self.avgpool(self.inception5b(self.inception5a(x))))

        x = torch.flatten(x, start_dim=1)

        return self.fc1(x)
