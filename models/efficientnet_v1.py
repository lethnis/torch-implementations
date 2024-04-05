from math import ceil
import torch
from torch import nn
from torchinfo import summary

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

params_dict = {
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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(in_channels // reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class InvertedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        expand_ratio,
        reduction=2,
        survival_prob=0.8,
    ):
        super().__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = ConvBlock(in_channels, hidden_dim, 3, 1, 1)

        self.conv = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim, kernel_size, stride, padding, hidden_dim),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def stochastic_depth(self, x):
        if not self.training:
            return x
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, x):
        in_x = x
        x = self.expand_conv(x) if self.expand else x

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + in_x
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super().__init__()
        width_factor, depth_factor, resolution, dropout_rate = version
        last_channels = ceil(1280 * width_factor)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes),
        )

    def create_features(self, width_factor, depth_factor, last_channels):

        channels = int(32 * width_factor)
        features = []
        features.append(ConvBlock(3, channels, 3, 2, 1))

        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride if layer == 0 else 1,
                        padding=kernel_size // 2,
                        expand_ratio=expand_ratio,
                    )
                )
                in_channels = out_channels

        features.append(ConvBlock(in_channels, last_channels, 1, 1, 0))
        return nn.Sequential(*features)

    def forward(self, x):
        x = self.avgpool(self.features(x))
        return self.classifier(torch.flatten(x, 1))


model = EfficientNet(params_dict["b0"], 1000)

x = torch.randn((1, 3, 224, 224))

print(model(x).shape)

summary(model, input_data=x)
