import torch
from torch import nn
from torchinfo import summary


blocks = [
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()

        expanded_channels = int(in_channels * expand_ratio)
        self.is_residual = in_channels == out_channels and stride == 1

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 3, stride, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.ReLU6(True),
                nn.Conv2d(expanded_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.ReLU6(True),
                nn.Conv2d(expanded_channels, expanded_channels, 3, stride, 1, groups=expanded_channels, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.ReLU6(True),
                nn.Conv2d(expanded_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        if self.is_residual:
            return x + self.conv(x)
        return self.conv(x)


class Mobilenetv2(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, alpha=1.0):
        super().__init__()

        features = [ConvBlock(in_channels, 32, 3, 2, 1)]
        in_channels = 32

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

        features.append(ConvBlock(in_channels, int(alpha * 1280), 1, 1, 0))

        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(int(alpha * 1280), num_classes)

    def forward(self, x):
        x = self.avgpool(self.features(x))
        return self.classifier(torch.flatten(x, 1))


model = Mobilenetv2(3, 10, 0.25)
x = torch.randn(1, 3, 224, 224)
print(model(x).shape)
summary(model, input_data=x, depth=5)
