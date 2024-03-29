import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.avg_pool2d(x, 2)
        return x


class DilatedCNN(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()

        self.conv = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=64, dilation=3),
            ConvBlock(in_channels=64, out_channels=128, dilation=3),
            ConvBlock(in_channels=128, out_channels=256, dilation=2),
            ConvBlock(in_channels=256, out_channels=512, dilation=2),
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            # nn.Linear(512, 128),
            # nn.PReLU(),
            # nn.BatchNorm1d(128),
            # nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.mean([2, 3])
        x = self.fc(x)
        return x

