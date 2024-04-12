import torch
from torch import nn
from torch.nn import functional as F
from torchview import draw_graph


class UNet(nn.Module):
    def __init__(self, features: list = [64, 128, 256, 512], in_channels: int = 3, out_channels: int = 1):
        """Implementation of a U-Net architecture.
        Original paper: https://arxiv.org/pdf/1505.04597.pdf.

        Differences:
            1. Same input shape and output shape.
            2. 3 input channels instead of 1.
            3. Added batchnorm after every conv.

        Args:
            features (list, optional): Number of channels and times image
                will be resized. Used both for contracting path and for
                expansive path. Defaults to [64, 128, 256, 512].
            in_channels (int, optional): number of image channels. Defaults to 3.
            out_channels (int, optional): number of classes to segment. Defaults to 1.
        """
        super().__init__()

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for feature in features:
            # add conv->batchnorm->relu->conv->batchnorm->relu
            # for default params number of in and out channels will be:
            # 3-64 -> 64-128 -> 128-256 -> 256-512
            # after every block there will be maxpool in forward method
            self.downs.append(self.get_conv_block(in_channels, feature))
            # rewrite in_channels for next iteration
            in_channels = feature

            # first will do convTranspose from previous block size
            # for convTranspose in-out channels will be 1024-512 -> 512-256 -> 256-128 -> 128-64
            # after convTranspose will do concatenate with downs and the conv block
            # for conv block in-out channels will be 1024-512 -> 512-256 -> 256-128 -> 128-64
            self.ups.insert(
                0,
                nn.ModuleList(
                    [nn.ConvTranspose2d(feature * 2, feature, 2, 2), self.get_conv_block(feature * 2, feature)]
                ),
            )

        # bottleneck after final down layer. ConvTranspose will get 1024 channels from here
        self.bottleneck = self.get_conv_block(features[-1], features[-1] * 2)

        # final conv for segmentation. Channels will be 64-1 for default parameters
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def get_conv_block(self, in_channels: int, out_channels: int):
        """Simple double conv block with batchnorm and relu."""

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):

        # remember out of every down block before maxpool
        skips = []
        for down in self.downs:
            x = down(x)
            skips.insert(0, x)
            x = F.max_pool2d(x, 2, 2)

        x = self.bottleneck(x)

        # first do convTranspose, then concatenate with output of corresponding
        # down block, then do conv block
        for skip, up in zip(skips, self.ups):
            x = up[0](x)
            x = torch.concat((skip, x), dim=1)
            x = up[1](x)

        return self.final_conv(x)


draw_graph(
    UNet(),
    torch.randn(1, 3, 224, 224),
    save_graph=True,
    expand_nested=True,
    depth=1,
    filename="U-Net expanded",
)

draw_graph(
    UNet(),
    torch.randn(1, 3, 224, 224),
    save_graph=True,
    expand_nested=True,
    depth=3,
    filename="U-Net collapsed",
)
