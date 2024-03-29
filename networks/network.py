import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
from util import *
import pretrainedmodels


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


def mobilenetv2(**kwargs):
    # model = models.MobileNetV2(**kwargs)

    # if pretrain is not None:
    #     checkpoint = torch.load(pretrain)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     print("=> loaded checkpoint {}, best_lwlrap: {:.4f} @ {}"
    #           .format(pretrain, checkpoint['best_lwlrap'], checkpoint['epoch']))
    model = models.mobilenet_v2(**kwargs)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, 80),
    )
    return model


def dpn98_(**kwargs):
    model = pretrainedmodels.models.dpn98(**kwargs)
    model.last_linear = nn.Conv2d(2688, 80, kernel_size=1, bias=True)
    return model
