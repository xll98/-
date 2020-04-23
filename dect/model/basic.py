import torch
import functools
import torch.nn as nn

def conv(inp, oup, kernel_size, stride, pad):

    conv = nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=pad, bias=True)
    nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
    nn.init.zeros_(conv.bias)

    return nn.Sequential(
        conv
    )

def conv_relu(inp, oup, kernel_size, stride, pad):

    conv = nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=pad, bias=True)
    nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
    nn.init.zeros_(conv.bias)

    return nn.Sequential(
        conv,
        nn.ReLU(inplace=True)
    )

def conv_bn_relu(inp, oup, kernel_size, stride, pad):

    conv = nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)
    nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')

    return nn.Sequential(
        conv,
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_bn(inp, oup, kernel_size, stride, pad):

    conv = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
    nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')

    return nn.Sequential(
        conv,
        nn.BatchNorm2d(oup)
    )

def dwconv(inp, oup, kernel_size, stride, pad):

    conv = nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=pad, bias=True, groups=inp)
    nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
    nn.init.zeros_(conv.bias)

    return nn.Sequential(
        conv
    )

def dwconv_bn(inp, oup, kernel_size, stride, pad):

    conv = nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=pad, bias=False, groups=inp)
    nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')

    return nn.Sequential(
        conv,
        nn.BatchNorm2d(oup)
    )

class FullConv(nn.Module):

    def __init__(self, in_channels, out_channels, feature_size):
        super(FullConv, self).__init__()

        self.conv_full = conv(in_channels, out_channels, kernel_size=feature_size, stride=1, pad=0)

    def forward(self, feature):

        feature = self.conv_full(feature)

        return feature

class DwFullConvRelu(nn.Module):

    def __init__(self, in_channels, out_channels, feature_size):
        super(DwFullConvRelu, self).__init__()

        # 分离全卷积, 输入 1 x c x h x w , 输出 1 x c x 1 x 1
        self.dw_conv_full = dwconv(in_channels, in_channels, kernel_size=feature_size, stride=1, pad=0)

        # 输入 1 x c x 1 x 1 , 输出 1 x c' x 1 x 1
        self.conv_1x1_relu = conv_relu(in_channels, out_channels, kernel_size=1, stride=1, pad=0)

    def forward(self, x):

        x = self.dw_conv_full(x)
        x = self.conv_1x1_relu(x)

        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, shuffle_groups=2):
        super().__init__()

        channels = in_channels // 2
        self.channels = channels

        self.conv1 = conv_bn_relu(channels, channels, kernel_size=1, stride=1, pad= 0)

        self.conv2 = dwconv_bn(channels, channels, kernel_size=5,  stride=1, pad=2)

        self.conv3 = conv_bn_relu(
            channels, channels, kernel_size=1, stride=1, pad= 0
        )

        self.shuffle = ShuffleBlock(shuffle_groups)

    def forward(self, x):
        x = x.contiguous()

        c = x.size(1) // 2

        x1 = x[:, :c, :, :]
        x2 = x[:, c:, :, :]
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)

        x = torch.cat((x1, x2), dim=1)

        x = self.shuffle(x)


        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shuffle_groups=2, **kwargs):
        super().__init__()
        channels = out_channels // 2

        self.conv11 = dwconv_bn(
            in_channels, in_channels, kernel_size=5, stride=2, pad= 2
        )

        self.conv12 = conv_bn_relu(
            in_channels, channels, kernel_size=1, stride=1, pad= 0
        )

        self.conv21 = conv_bn_relu(
            in_channels, channels, kernel_size=1, stride=1, pad= 0
        )

        self.conv22 = dwconv_bn(
            channels, channels, kernel_size=5, stride=2, pad= 2
        )

        self.conv23 = conv_bn_relu(
            channels, channels, kernel_size=1, stride=1, pad= 0
        )

        self.shuffle = ShuffleBlock(shuffle_groups)

    def forward(self, x):
        x1 = self.conv11(x)

        x1 = self.conv12(x1)

        x2 = self.conv21(x)
        x2 = self.conv22(x2)
        x2 = self.conv23(x2)

        x = torch.cat((x1, x2), dim=1)
        x = self.shuffle(x)
        return x


def channel_shuffle(x, g):
    n, c, h, w = x.size()
    x = x.view(n, g, c // g, h, w).permute(
        0, 2, 1, 3, 4).contiguous().view(n, c, h, w)
    return x


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, g=self.groups)