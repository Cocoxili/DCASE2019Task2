import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


def resnet18(pretrained=False, **kwargs):
    model = models.resnet18(pretrained=pretrained)
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
    #                            bias=False)
    # model.avgpool = nn.AvgPool2d((2, 7), stride=(2, 7))
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(512, 80)
    return model


def resnet50(pretrained=False, **kwargs):
    model = models.resnet50(pretrained=pretrained)
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
    #                            bias=False)
    # model.avgpool = nn.AvgPool2d((2, 7), stride=(2, 7))
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(512 * 4, 80)
    return model


def resnet101_mfcc(pretrained=False, **kwargs):
    model = models.resnet101(pretrained=pretrained)
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
    #                            bias=False)
    model.avgpool = nn.AvgPool2d((2, 5), stride=(2, 5))
    model.fc = nn.Linear(512 * 4, 80)
    return model


def densenet121(**kwargs):
    model = models.densenet121(**kwargs)
    return model


def vgg11(**kwargs):
    model = models.vgg11_bn(**kwargs)
    return model
