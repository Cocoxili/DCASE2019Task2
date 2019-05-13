import torch
import torch.nn as nn


def BCEWithLogits_(input, target, size_average=True):
    assert input.size() == target.size()

    sig = nn.Sigmoid()
    l_n = (target * torch.log(sig(input)) + (1-target) * torch.log(1-sig(input)))

    if size_average:
        return -torch.mean(l_n)
    else:
        return -torch.sum(l_n)


def SoftBCE(input, target, beta, size_average=True):
    assert input.size() == target.size()
    eps = 1e-12
    xs = nn.Sigmoid()(input)
    l_n = target * torch.log(xs+eps) + (1-target) * torch.log(1-xs+eps)
    l_n = beta * l_n + (1-beta) * xs * torch.log(xs+eps)
    if size_average:
        return -torch.mean(l_n)
    else:
        return -torch.sum(l_n)


def Q_BCE(input, target, q):
    assert input.size() == target.size()
    # xs = nn.Sigmoid()(input)
    xs = nn.Softmax(dim=1)(input)
    return (1 - (torch.max(xs * target)) ** q) / q
    # return (1 - (torch.sum(xs * target)/torch.sum(target)) ** q) / q


if __name__ == '__main__':
    loss = nn.BCEWithLogitsLoss()
    input = torch.randn((4, 3), requires_grad=True)
    target = torch.empty((4, 3)).random_(2)
    print(input)
    print(target)
    print(input.size(), target.size())
    output = loss(input, target)
    print("output:", output)
    # print(output.backward())

    output2 = BCEWithLogits_(input, target, size_average=True)
    print('BCEwithLogits_:', output2)

    output3 = SoftBCE(input, target, beta=0.3)
    print('SoftBCE:', output3)

    output4 = Q_BCE(input, target, q=0.5)
    print('Q_BCE:', output4)