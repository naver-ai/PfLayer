"""
"Learning Features from Parameter-Free Layers"
Copyright (c) 2022-present NAVER Corp.
Apache-2.0
"""

# This code is written upon https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1, padding=None):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding,
                      groups=groups, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
        )


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1, padding=None,
                 act_layer=nn.ReLU):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding,
                      groups=groups, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
            act_layer(inplace=True) if act_layer is not None else None,
        )


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
    ) -> None:
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        self.reduction_block = ConvBNReLU(inplanes, width, kernel_size=1)

        self.conv_block = ConvBNReLU(width, width, kernel_size=3, stride=stride,
                                     groups=groups, dilation=dilation)

        self.expansion_block = ConvBN(width, planes * self.expansion, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.reduction_block(x)
        out = self.conv_block(out)
        out = self.expansion_block(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MaxBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
    ) -> None:
        super(MaxBottleneck, self).__init__()

        width = int(planes * (base_width / 64.)) * groups

        self.reduction_block = ConvBNReLU(inplanes, width, kernel_size=1)
        self.conv_block = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        self.expansion_block = ConvBN(width, planes * self.expansion, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.reduction_block(x)
        out = self.conv_block(out)

        out = self.expansion_block(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class EffBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
    ) -> None:
        super(EffBottleneck, self).__init__()

        width = int(planes * (base_width / 64.)) * groups

        if stride == 2:
            self.reduction_block = ConvBNReLU(inplanes, width, kernel_size=1)
            self.conv_block = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2),
                                            ConvBNReLU(width, width, kernel_size=3, stride=1, groups=groups,
                                                       dilation=dilation))
        else:
            self.reduction_block = ConvBNReLU(inplanes, width, kernel_size=1)
            self.conv_block = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)

        self.expansion_block = ConvBN(width, planes * self.expansion, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.reduction_block(x)
        out = self.conv_block(out)

        out = self.expansion_block(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[Bottleneck, MaxBottleneck, EffBottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            width_mult: float = 1.0,

    ) -> None:
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.block_idx = 0
        self.net_block_idx = sum(layers)

        self.conv1 = ConvBNReLU(3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, int(64 * width_mult), layers[0], stride=1, dilate=False)
        self.layer2 = self._make_layer(block, int(128 * width_mult), layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(256 * width_mult), layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(512 * width_mult), layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512 * width_mult) * block.expansion, num_classes)

    def _make_layer(self, block: Type[Union[Bottleneck, MaxBottleneck, EffBottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False,
                    ) -> nn.Sequential:
        downsample = None
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = ConvBN(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride)

        layers = []
        for block_idx in range(blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            layers.append(block(self.inplanes, planes, groups=self.groups, stride=stride, downsample=downsample,
                                base_width=self.base_width, dilation=self.dilation))
            self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet50(pretrained=False, layer_setting=[3, 4, 6, 3], width_mult: float = 1.0, **kwargs: Any) -> ResNet:
    return ResNet(Bottleneck, layers=layer_setting, width_mult=width_mult, **kwargs)


def resnet50_max(pretrained=False, layer_setting=[3, 4, 6, 3], width_mult: float = 1.0, **kwargs: Any) -> ResNet:
    return ResNet(MaxBottleneck, layers=layer_setting, width_mult=width_mult, **kwargs)


def resnet50_hybrid(pretrained=False, layer_setting=[3, 4, 6, 3], width_mult: float = 1.0, **kwargs: Any) -> ResNet:
    return ResNet(EffBottleneck, layers=layer_setting, width_mult=width_mult, **kwargs)


if __name__ == '__main__':
    # model = resnet50_max().eval()  # .cuda()
    model = resnet50_hybrid().eval()  # .cuda()

    input = torch.randn(1, 3, 224, 224)  # .cuda()
    output = model(input)

    loss = output.sum()
    loss.backward()
    print('Checked a single forward/backward iteration')
