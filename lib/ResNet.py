import torch
import torch.nn as nn

from torchsummary import summary
from functools import partial
from dataclasses import dataclass
from collections import OrderedDict


def Conv3x3(in_channels, out_channels, stride=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=dilation, bias=False, dilation=dilation)


def Conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, bias=False)


def ConvBn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({
        'conv': conv(in_channels, out_channels, *args, **kwargs),
        'bn': nn.BatchNorm2d(out_channels)
    }))


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.block = nn.Identity()
        self.shortcut = nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        # If the dimension of input and output is not the same, we need to use a 1d conv
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.block(x)
        x += residual
        x = self.activation(x)
        return x


class ResNetResidualBlock(ResidualBlock):

    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling = expansion, downsampling
        self.shortcut = nn.Sequential(OrderedDict({
            'conv': Conv1x1(self.in_channels, self.expanded_channels, stride=self.downsampling),
            'bn': nn.BatchNorm2d(self.expanded_channels)
        })) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.block = nn.Sequential(
            ConvBn(self.in_channels, self.out_channels, Conv3x3, stride=self.downsampling),
            activation(),
            ConvBn(self.out_channels, self.expanded_channels, Conv3x3)
        )


class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.block = nn.Sequential(
            ConvBn(self.in_channels, self.out_channels, Conv1x1),
            activation(),
            ConvBn(self.out_channels, self.out_channels, Conv3x3, stride=self.downsampling),
            activation(),
            ConvBn(self.out_channels, self.expanded_channels, Conv1x1)
        )


class ResNetLayer(nn.Module):

    def __init__(self, in_channels, out_channels, block=ResNetBottleNeckBlock, n=1, *args, **kwargs):
        super().__init__()
        # Perform downsampling by setting the stride to 2
        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetEncoder(nn.Module):

    def __init__(self, in_channels=3, block_sizes=[64, 128, 256, 512], depths=[3, 4, 6, 3],
                 activation=nn.ReLU, block=ResNetBottleNeckBlock, *args, **kwargs):
        super().__init__()

        self.block_sizes = block_sizes

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.block_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.block_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_block_sizes = list(zip(block_sizes, block_sizes[1:]))

        self.blocks = nn.ModuleList([
            ResNetLayer(block_sizes[0], block_sizes[0], n=depths[0], activation=activation,
                        *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]
        ])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResNetDecoder(nn.Module):

    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


class ResNet(nn.Module):

    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResNetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def ResNet34(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, depths=[3, 4, 6, 3])


def ResNet50(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, depths=[3, 4, 6, 3])


def ResNet101(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 23, 3])
