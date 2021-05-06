# from https://github.com/codyaustun/pytorch-resnet/blob/master/resnet/cifar10/models/resnet.py

import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from architectures.memory import MemoryWrapLayer,EncoderMemoryWrapLayer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or inplanes != (planes * self.expansion):
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
        else:
            self.shortcut = nn.Sequential()
    
    def forward(self, inputs):
        H = F.relu(self.bn1(self.conv1(inputs)))

        H = self.bn2(self.conv2(H))

        H += self.shortcut(inputs)
        outputs = F.relu(H)

        return outputs



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        if stride != 1 or inplanes != (planes * self.expansion):
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        H = F.relu(self.bn1(self.conv1(inputs)))

        H = F.relu(self.bn2(self.conv2(H)))

        H = self.bn3(self.conv3(H))

        H += self.shortcut(inputs)
        outputs = F.relu(H)

        return outputs


class ResNet(nn.Module):

    def __init__(self, Block, layers, filters, num_classes=10, inplanes=None):
        self.inplanes = inplanes or filters[0]
        super().__init__()

        self.pre_act = 'Pre' in Block.__name__

        self.conv1 = nn.Conv2d(3, self.inplanes, 3, padding=1, bias=False)
        if not self.pre_act:
            self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.num_sections = len(layers)
        for section_index, (size, planes) in enumerate(zip(layers, filters)):
            section = []
            for layer_index in range(size):
                if section_index != 0 and layer_index == 0:
                    stride = 2
                else:
                    stride = 1
                section.append(Block(self.inplanes, planes, stride=stride))
                self.inplanes = planes * Block.expansion
            section = nn.Sequential(*section)
            setattr(self, f'section_{section_index}', section)
        if self.pre_act:
            self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.linear = nn.Linear(filters[-1] * Block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        H = self.conv1(inputs)
        H = self.bn1(H)
        H = F.relu(H)

        for section_index in range(self.num_sections):
            H = getattr(self, f'section_{section_index}')(H)

        H = F.avg_pool2d(H, H.size()[2:])

        H = H.view(H.size(0), -1)
        outputs = self.linear(H)


        return outputs

class MemoryResNet(nn.Module):

    def __init__(self, Block, layers, filters, num_classes=10, inplanes=None):
        self.inplanes = inplanes or filters[0]
        super().__init__()

        self.pre_act = 'Pre' in Block.__name__

        self.conv1 = nn.Conv2d(3, self.inplanes, 3, padding=1, bias=False)
        if not self.pre_act:
            self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.num_sections = len(layers)
        for section_index, (size, planes) in enumerate(zip(layers, filters)):
            section = []
            for layer_index in range(size):
                if section_index != 0 and layer_index == 0:
                    stride = 2
                else:
                    stride = 1
                section.append(Block(self.inplanes, planes, stride=stride))
                self.inplanes = planes * Block.expansion
            section = nn.Sequential(*section)
            setattr(self, f'section_{section_index}', section)
        if self.pre_act:
            self.bn1 = nn.BatchNorm2d(self.inplanes)

        #self.linear = nn.Linear(filters[-1] * Block.expansion, num_classes)
        
        self.mw = MemoryWrapLayer(filters[-1] * Block.expansion,num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward_encoder(self, inputs):
        H = self.conv1(inputs)
        H = self.bn1(H)
        H = F.relu(H)

        for section_index in range(self.num_sections):
            H = getattr(self, f'section_{section_index}')(H)

        H = F.avg_pool2d(H, H.size()[2:])

        H = H.view(H.size(0), -1)
        #outputs = self.linear(H)
        return H
    
    def forward(self, x, ss,return_weights=False):

        #input
        out = self.forward_encoder(x)
        out_ss = self.forward_encoder(ss)

        # prediction
        out_mw = self.mw(out,out_ss,return_weights)
        return out_mw


class EncoderMemoryResNet(MemoryResNet):

     def __init__(self, Block, layers, filters, num_classes=10, inplanes=None):
        self.inplanes = inplanes or filters[0]
        super(MemoryResNet,self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.num_sections = len(layers)
        for section_index, (size, planes) in enumerate(zip(layers, filters)):
            section = []
            for layer_index in range(size):
                if section_index != 0 and layer_index == 0:
                    stride = 2
                else:
                    stride = 1
                section.append(Block(self.inplanes, planes, stride=stride))
                self.inplanes = planes * Block.expansion
            section = nn.Sequential(*section)
            setattr(self, f'section_{section_index}', section)

        #self.linear = nn.Linear(filters[-1] * Block.expansion, num_classes)
        
        self.mw = EncoderMemoryWrapLayer(filters[-1] * Block.expansion,num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# From "Deep Residual Learning for Image Recognition"
def ResNet20(num_classes=10):
    return ResNet(BasicBlock, layers=[3] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def ResNet32(num_classes=10):
    return ResNet(BasicBlock, layers=[5] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def ResNet44(num_classes=10):
    return ResNet(BasicBlock, layers=[7] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def ResNet56(num_classes=10):
    return ResNet(BasicBlock, layers=[9] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def ResNet110(num_classes=10):
    return ResNet(BasicBlock, layers=[18] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def ResNet1202(num_classes=10):
    return ResNet(BasicBlock, layers=[200] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)



# From "Deep Networks with Stochastic Depth" for SVHN Experiments
def ResNet152SVHN(num_classes=10):
    return ResNet(BasicBlock, layers=[25] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)

# From kunagliu/pytorch
def ResNet18(num_classes=10):
    return ResNet(BasicBlock, layers=[2, 2, 2, 2], filters=[64, 128, 256, 512],
                  num_classes=num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, layers=[3, 4, 6, 3], filters=[64, 128, 256, 512],
                  num_classes=num_classes)


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, layers=[3, 4, 6, 3], filters=[64, 128, 256, 512],
                  num_classes=num_classes)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck,
                  layers=[3, 4, 23, 3], filters=[64, 128, 256, 512],
                  num_classes=num_classes)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck,
                  layers=[3, 8, 36, 3], filters=[64, 128, 256, 512],
                  num_classes=num_classes)

# From "Deep Residual Learning for Image Recognition"
def MemoryResNet20(num_classes=10):
    return MemoryResNet(BasicBlock, layers=[3] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def MemoryResNet32(num_classes=10):
    return MemoryResNet(BasicBlock, layers=[5] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def MemoryResNet44(num_classes=10):
    return MemoryResNet(BasicBlock, layers=[7] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def MemoryResNet56(num_classes=10):
    return MemoryResNet(BasicBlock, layers=[9] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def MemoryResNet110(num_classes=10):
    return MemoryResNet(BasicBlock, layers=[18] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def MemoryResNet1202(num_classes=10):
    return MemoryResNet(BasicBlock, layers=[200] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)



# From "Deep Networks with Stochastic Depth" for SVHN Experiments
def MemoryResNet152SVHN(num_classes=10):
    return MemoryResNet(BasicBlock, layers=[25] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)

# From kunagliu/pytorch
def MemoryResNet18(num_classes=10):
    return MemoryResNet(BasicBlock, layers=[2, 2, 2, 2], filters=[64, 128, 256, 512],
                  num_classes=num_classes)

def MemoryResNet34(num_classes=10):
    return MemoryResNet(BasicBlock, layers=[3, 4, 6, 3], filters=[64, 128, 256, 512],
                  num_classes=num_classes)


def MemoryResNet50(num_classes=10):
    return MemoryResNet(Bottleneck, layers=[3, 4, 6, 3], filters=[64, 128, 256, 512],
                  num_classes=num_classes)


def MemoryResNet101(num_classes=10):
    return MemoryResNet(Bottleneck,
                  layers=[3, 4, 23, 3], filters=[64, 128, 256, 512],
                  num_classes=num_classes)


def MemoryResNet152(num_classes=10):
    return MemoryResNet(Bottleneck,
                  layers=[3, 8, 36, 3], filters=[64, 128, 256, 512])


# From "Deep Residual Learning for Image Recognition"
def EncoderMemoryResNet20(num_classes=10):
    return EncoderMemoryResNet(BasicBlock, layers=[3] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def EncoderMemoryResNet32(num_classes=10):
    return EncoderMemoryResNet(BasicBlock, layers=[5] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def EncoderMemoryResNet44(num_classes=10):
    return EncoderMemoryResNet(BasicBlock, layers=[7] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def EncoderMemoryResNet56(num_classes=10):
    return EncoderMemoryResNet(BasicBlock, layers=[9] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def EncoderMemoryResNet110(num_classes=10):
    return EncoderMemoryResNet(BasicBlock, layers=[18] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def EncoderMemoryResNet1202(num_classes=10):
    return EncoderMemoryResNet(BasicBlock, layers=[200] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)



# From "Deep Networks with Stochastic Depth" for SVHN Experiments
def EncoderMemoryResNet152SVHN(num_classes=10):
    return EncoderMemoryResNet(BasicBlock, layers=[25] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)

# From kunagliu/pytorch
def EncoderMemoryResNet18(num_classes=10):
    return EncoderMemoryResNet(BasicBlock, layers=[2, 2, 2, 2], filters=[64, 128, 256, 512],
                  num_classes=num_classes)

def EncoderMemoryResNet34(num_classes=10):
    return EncoderMemoryResNet(BasicBlock, layers=[3, 4, 6, 3], filters=[64, 128, 256, 512],
                  num_classes=num_classes)


def EncoderMemoryResNet50(num_classes=10):
    return EncoderMemoryResNet(Bottleneck, layers=[3, 4, 6, 3], filters=[64, 128, 256, 512],
                  num_classes=num_classes)


def EncoderMemoryResNet101(num_classes=10):
    return EncoderMemoryResNet(Bottleneck,
                  layers=[3, 4, 23, 3], filters=[64, 128, 256, 512],
                  num_classes=num_classes)


def EncoderMemoryResNet152(num_classes=10):
    return EncoderMemoryResNet(Bottleneck,
                  layers=[3, 8, 36, 3], filters=[64, 128, 256, 512],
                  num_classes=num_classes)