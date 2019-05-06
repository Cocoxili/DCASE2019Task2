
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class TestCNN(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()

        feature = nn.Sequential(
            ConvBNReLU(3, 64, 3, 1, 1),
            ConvBNReLU(3, 64, 3, 1, 1),
            ConvBNReLU(3, 64, 3, 1, 1),
            ConvBNReLU(3, 64, 3, 1, 1),
            ConvBNReLU(3, 64, 3, 1, 1),
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            # nn.Linear(512, 128),
            # nn.PReLU(),
            # nn.BatchNorm1d(128),
            # nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean([2, 3])
        x = self.fc(x)
        return x

