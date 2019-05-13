
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


class BNReLUConv(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(BNReLUConv, self).__init__(
            nn.BatchNorm2d(in_planes),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False)
        )


class TestCNN(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()

        self.feature = nn.Sequential(
            BNReLUConv(3, 64, 3),
            BNReLUConv(64, 64, 3),
            BNReLUConv(64, 128, 3),
            BNReLUConv(128, 128, 3),
            BNReLUConv(128, 256, 3),
            BNReLUConv(256, 256, 3),
            BNReLUConv(256, 512, 3),
            BNReLUConv(512, 512, 3),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.PReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
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
        x = self.feature(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


class TestCNN2(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        # padding = (kernel_size - 1) // 2
        self.feature = nn.Sequential(nn.BatchNorm2d(3),
                      nn.ReLU(),
                      nn.Conv2d(3, 24, (5, 5), 1, 2),
                      nn.BatchNorm2d(24),
                      nn.ReLU(),
                      nn.MaxPool2d((4, 2)),

                      nn.BatchNorm2d(24),
                      nn.ReLU(),
                      nn.Conv2d(24, 48, 5, 1, 2),
                      nn.BatchNorm2d(48),
                      nn.ReLU(),
                      nn.MaxPool2d((4, 2)),

                      nn.BatchNorm2d(48),
                      nn.ReLU(),
                      nn.Conv2d(48, 48, 5, 1, 2),
                      nn.BatchNorm2d(48),
                      nn.ReLU()
                      )
        self.classifier = nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(14208, 64),
                                        nn.Dropout(0.5),
                                        nn.Linear(64, 80))
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
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.classifier(x)
        return x
