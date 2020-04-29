import torchvision
import torch.nn.functional as F
from torch import nn
from models.functions import activation_func


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channel, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        # convolve separately each channel of specified kernel
        self.depth_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        # increase the number of channels of each feature
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channel, kernel_size=(1, 1),
                                   stride=(1, 1), padding=0, dilation=1, groups=1, bias=False)

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.pointwise(out)
        return out


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, repeats, strides=1, act_type='relu', activation_first=True,
                 grow_first=True):
        super().__init__()

        # define skip connection
        if in_channels != out_channels or strides != 1:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides, bias=False)
            self.skip_bn = nn.BatchNorm2d(out_channels)
        else:
            self.skip_connection = None

        self.activation = activation_func(act_type)

        layers = []
        channels = in_channels

        if grow_first:
            layers.append(self.activation)
            layers.append(DepthwiseSeparableConv(in_channels, out_channels, 3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            channels = out_channels

        for i in range(repeats):
            layers.append(self.activation)
            layers.append(DepthwiseSeparableConv(channels, channels, 3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(channels))

        if not grow_first:
            layers.append(self.activation)
            layers.append(DepthwiseSeparableConv(in_channels, out_channels, 3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))

        if not activation_first:
            layers = layers[1:]
        else:
            layers[0] = activation_func(act_type)

        if strides != 1:
            layers.append(nn.MaxPool2d(3, strides, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)

        if self.skip_connection is not None:
            skip = self.skip_connection(x)
            skip = self.skip_bn(skip)
        else:
            skip = x

        out += skip
        return out


class Head(nn.Module):
    """Pre-trained layers on ImageNet from vgg19"""
    def __init__(self, pretrained=True):
        super().__init__()
        features = list(torchvision.models.vgg19(pretrained=pretrained, progress=True).features)[:3]
        self.features = nn.Sequential(*features)

    def forward(self, x):
        return self.features(x)


class XrayRSCNN(nn.Module):

    def __init__(self, num_classes=2, act_type='relu'):
        super().__init__()
        self.num_classes = num_classes

        self.head = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2)
        )

        self.block1 = Block(32, 64, 1, 2, activation_first=False, grow_first=True)
        self.block2 = Block(64, 128, 1, 2, activation_first=True, grow_first=True)
        self.block3 = Block(128, 256, 1, 2, activation_first=True, grow_first=False)

        self.conv_1 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.act_1 = activation_func(act_type)
        self.conv_2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.act_2 = activation_func(act_type)
        self.bn_1 = nn.BatchNorm2d(256)
        self.maxpool_1 = nn.MaxPool2d(2, 2)

        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes)
        )

        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.head(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.conv_1(x)
        x = self.act_1(x)
        x = self.conv_2(x)
        x = self.act_2(x)
        x = self.bn_1(x)
        x = self.maxpool_1(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def inference(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x


class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.act1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.act2 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.act3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.act4 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool4 = nn.MaxPool2d(2, 2)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn1(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.act4(x)
        x = self.bn2(x)
        x = self.maxpool4(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def inference(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x


def get_resnet_50_test(num_class=2, pretrained=True):
    model_ft = torchvision.models.resnet50(pretrained=pretrained)

    for param in model_ft.parameters():
        param.requires_grad = False

    num_features = model_ft.fc.in_features

    model_ft.fc = nn.Sequential(
        nn.Dropout(0.15),
        nn.Linear(num_features, num_class)
    )

    return model_ft
