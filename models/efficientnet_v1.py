import torch
from torch import nn

from ._options import EfficientNet_options

# structure of MBblocks as described in the paper
base_model = [
    # expand ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

# settings for a specific model
models_dict: dict[EfficientNet_options, list] = {
    # (width_factor, depth_factor, resolution, dropout_rate)
    "b0": (1.0, 1.0, 224, 0.2),
    "b1": (1.0, 1.1, 240, 0.2),
    "b2": (1.1, 1.2, 260, 0.3),
    "b3": (1.2, 1.4, 300, 0.3),
    "b4": (1.4, 1.8, 380, 0.4),
    "b5": (1.6, 2.2, 456, 0.4),
    "b6": (1.8, 2.6, 528, 0.5),
    "b7": (2.0, 3.1, 600, 0.5),
}


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
        act: bool = True,
        bias: bool = False,
    ):
        """Simple conv block with conv -> batchnorm -> silu

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (int): kernel size.
            stride (int): stride.
            groups (int, optional): number of groups. Needed for separable conv. Defaults to 1.
            act (bool, optional): apply activation or don't. Defaults to True.
            bias (bool, optional): apply bias or don't. Defaults to False.
        """
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        # if no activation just apply Identity (the same)
        self.silu = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels: int, reduce_ratio: int = 24):
        """Squeeze and Excitation block. Applies channel-wise attention to input channels.
        avgpool -> conv -> silu -> conv -> sigmoid. Applies weights to each channels.

        Args:
            in_channels (int): number of input channels.
            reduce_ratio (int, optional): multiplier for intermediate channels. Defaults to 24.
        """
        super().__init__()
        reduced_channels = in_channels // reduce_ratio
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, reduced_channels, 1)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(reduced_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.conv2(self.silu(self.conv1(self.avgpool(x)))))


class MBBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int, expansion_ratio: int, reduce_ratio=24
    ):
        """Inverted residual block with linear bottleneck. The same as in MobileNetv3.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (int): kernel size.
            stride (int): stride.
            expansion_ratio (int): multiplier for bottleneck expansion channels.
            reduce_ratio (int, optional): . Defaults to 24.
        """
        super().__init__()

        exp_channels = in_channels * expansion_ratio
        self.add = in_channels == out_channels and stride == 1

        # point-wise conv to increase channels
        self.conv1 = ConvBlock(in_channels, exp_channels, 1, 1)
        # depth-wise conv to extract features
        self.conv2 = ConvBlock(exp_channels, exp_channels, kernel_size, stride, groups=exp_channels)
        # get weighted channels
        self.se = SqueezeExcitation(exp_channels, reduce_ratio)
        # point-wise conv to reduce channels. No activation!
        self.conv3 = ConvBlock(exp_channels, out_channels, 1, 1, act=False)

    def forward(self, inputs):
        x = self.conv3(self.se(self.conv2(self.conv1(inputs))))
        if self.add:
            return inputs + x
        return x


class Classifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout_rate: float):
        """Classifier head for the model. Gets final features, applies avgpool and dropout.

        Args:
            in_channels (int): number of input channels.
            num_classes (int): number of classes.
            dropout_rate (float): dropout rate.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        return self.fc(self.dropout(torch.flatten(self.pool(x), 1)))


class EfficientNetv1(nn.Module):
    def __init__(
        self,
        width_factor: float,
        depth_factor: float,
        resolution: int = 224,
        dropout_rate: float = 0.5,
        in_channels: int = 3,
        num_classes: int = 1000,
    ):
        """EffitientNetv1 as described in the original paper.
        Original paper: https://arxiv.org/pdf/1905.11946.pdf.
        Paper walkthrough and implementation: https://www.youtube.com/watch?v=eFMmqjDbcvw.

        Args:
            width_factor (float): multiplier for number of channels.
            depth_factor (float): multiplier for number of layers.
            resolution (int, optional): one side of an image. Needed to sanity check. Defaults to 224.
            dropout_rate (float, optional): dropout rate. Defaults to 0.5.
            in_channels (int, optional): number of input channels (image channels). Defaults to 3.
            num_classes (int, optional): number of classes. Defaults to 1000.
        """
        super().__init__()

        # input size to compare with inputs in forward method
        self.input_size = torch.Size((in_channels, resolution, resolution))

        # add first layer as described in paper
        features = [ConvBlock(in_channels, 32, 3, 2)]
        in_channels = 32

        # for every block in the base model
        for exp_ratio, out_channels, repeats, stride, kernel_size in base_model:
            out_channels = int(width_factor * out_channels)
            repeats = max(int(depth_factor * repeats), 1)
            for i in range(repeats):
                # stride will be applied only in first iteration
                features.append(MBBlock(in_channels, out_channels, kernel_size, stride if i == 0 else 1, exp_ratio))
                in_channels = out_channels

        # add last channel
        last_channels = int(width_factor * 1280)
        features.append(nn.Conv2d(out_channels, last_channels, 1))

        self.features = nn.Sequential(*features)

        # add classifier
        self.classifier = Classifier(last_channels, num_classes, dropout_rate)

    def forward(self, x):
        # compare shapes
        assert x.shape[1:] == self.input_size, f"Required image size: {self.input_size}. Got {x.shape[1:]}"
        return self.classifier(self.features(x))

    @classmethod
    def from_options(cls, model_name: EfficientNet_options, in_channels=3, num_classes=1000):
        return EfficientNetv1(*models_dict[model_name], in_channels=in_channels, num_classes=num_classes)
