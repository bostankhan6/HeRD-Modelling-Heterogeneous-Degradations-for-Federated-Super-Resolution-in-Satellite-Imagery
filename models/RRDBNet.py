import os
from typing import Any, cast, Dict, List, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor

class RRDBNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            growth_channels: int = 32,
            num_rrdb: int = 23,
            upscale: int = 4,
    ) -> None:
        super(RRDBNet, self).__init__()
        self.upscale = upscale

        # The first layer of convolutional layer.
        self.conv1 = nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1))

        # Feature extraction backbone network.
        trunk = []
        for _ in range(num_rrdb):
            trunk.append(_ResidualResidualDenseBlock(channels, growth_channels))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks.
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))

        # Upsampling convolutional layer.
        if upscale == 2:
            self.upsampling1 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True)
            )
        if upscale == 4:
            self.upsampling1 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True)
            )
            self.upsampling2 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True)
            )

        if upscale == 8:
            self.upsampling1 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True)
            )
            self.upsampling2 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True)
            )
            self.upsampling3 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True)
            )
            
        if upscale == 6:
            self.upsampling1 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True)
            )
            self.upsampling2 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True)
            )

        # Reconnect a layer of convolution block after upsampling.
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Output layer.
        self.conv4 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

        # Initialize all layer
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.2
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)

        if self.upscale == 2:
            x = self.upsampling1(F_torch.interpolate(x, scale_factor=2, mode="nearest"))
        if self.upscale == 4:
            x = self.upsampling1(F_torch.interpolate(x, scale_factor=2, mode="nearest"))
            x = self.upsampling2(F_torch.interpolate(x, scale_factor=2, mode="nearest"))
        if self.upscale == 6:
            x = self.upsampling1(F_torch.interpolate(x, scale_factor=2, mode="nearest"))
            x = self.upsampling2(F_torch.interpolate(x, scale_factor=3, mode="nearest"))
        if self.upscale == 8:
            x = self.upsampling1(F_torch.interpolate(x, scale_factor=2, mode="nearest"))
            x = self.upsampling2(F_torch.interpolate(x, scale_factor=2, mode="nearest"))
            x = self.upsampling3(F_torch.interpolate(x, scale_factor=2, mode="nearest"))

        x = self.conv3(x)
        x = self.conv4(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class _ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels + growth_channels * 0, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(channels + growth_channels * 1, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(channels + growth_channels * 2, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(channels + growth_channels * 3, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(channels + growth_channels * 4, channels, (3, 3), (1, 1), (1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))

        x = torch.mul(out5, 0.2)
        x = torch.add(x, identity)

        return x


class _ResidualResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_ResidualResidualDenseBlock, self).__init__()
        self.rdb1 = _ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = _ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = _ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        x = self.rdb1(x)
        x = self.rdb2(x)
        x = self.rdb3(x)

        x = torch.mul(x, 0.2)
        x = torch.add(x, identity)

        return x
    
def rrdbnet_x4(**kwargs: Any) -> RRDBNet:
    model = RRDBNet(upscale=4, **kwargs)

    return model