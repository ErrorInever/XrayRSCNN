from torch import nn


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=False)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=False)],
        ['selu', nn.SELU(inplace=False)],
        ['none', nn.Identity()]
    ])[activation]
