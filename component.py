from numpy import pad
import torch.nn as nn
import torch.nn.functional as F
from utils.functions import position_coding
import torch
import os
import torch
import torch.nn as nn

# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.models.layers import DropPath, trunc_normal_
# from timm.models.registry import register_model
# from timm.models.layers.helpers import to_2tuple

import math
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size=3, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d((2, 2)),
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size=3, stride=1):
        super(BasicBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1),
            nn.Conv2d(planes, planes, kernel_size, stride=1, padding=0, groups=planes, bias=False),
            nn.BatchNorm2d(planes),
        )
        
        # if stride != 1 or in_planes != planes:
        #     self.shortcut = nn.Sequential(
        #         # nn.AvgPool2d((2, 2)),
        #         nn.Conv2d(in_planes, self.expansion * planes, kernel_size=(kernel_size - 1) * 2, stride=1, bias=False, groups=in_planes, padding=0),
        #         nn.BatchNorm2d(self.expansion * planes)
        #     )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class normal_block2(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(in_ch)

        self.shortcut = nn.Sequential()
    
    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = F.gelu(self.bn2(self.conv2(out)) + self.shortcut(x))
        return out


class normal_block(nn.Module):
    def __init__(self, in_ch, middle_ch) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, middle_ch, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_ch, middle_ch, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(2 * middle_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch)
        # self.relu1 = relu1
        # self.relu2 = relu2
        # self.linear = nn.Conv2d(bn_ch, bn_ch, kernel_size=1, stride=1, padding=0)
        # if norm1:
        # self.bn1 = nn.BatchNorm2d(bn_ch)
        # self.bn2 = nn.BatchNorm2d(bn_ch)
        self.bn4 = nn.BatchNorm2d(in_ch)
        self.bn1 = nn.Sequential()
        self.bn2 = nn.Sequential()
        self.bn3 = nn.BatchNorm2d(in_ch)
        # else:
        #     self.bn1 = nn.Sequential()
        # self.pos_code1 = pos_code1
        # self.pos_code2 = pos_code2
        
        # if norm2:
        #     self.bn2 = nn.BatchNorm2d(bn_ch)
        # else:
        #     self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out2 = self.conv2(x)
        out2 = self.bn2(out2)
        out = torch.cat([out, out2], 1)
        out = F.gelu(self.bn4(self.conv3(out)))
        out = F.gelu(self.bn3(self.conv4(out)) + self.shortcut(x))
        return out

class shuffle_block(nn.Module):
    def __init__(self, in_ch1, out_ch1, bn_ch) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch1, out_ch1, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_ch1, out_ch1, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(2 * bn_ch, bn_ch, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(2* bn_ch, 2* bn_ch, 3, 1, 1, groups=2 * bn_ch)
        # self.relu1 = relu1
        # self.relu2 = relu2
        # self.linear = nn.Conv2d(bn_ch, bn_ch, kernel_size=1, stride=1, padding=0)
        # if norm1:
        # self.bn1 = nn.BatchNorm2d(bn_ch)
        # self.bn2 = nn.BatchNorm2d(bn_ch)
        self.bn3 = nn.BatchNorm2d(bn_ch)
        self.bn1 = nn.Sequential()
        self.bn2 = nn.Sequential()
        # else:
        #     self.bn1 = nn.Sequential()
        # self.pos_code1 = pos_code1
        # self.pos_code2 = pos_code2
        
        # if norm2:
        #     self.bn2 = nn.BatchNorm2d(bn_ch)
        # else:
        #     self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
    
    def forward(self, x):
        # shortcut = self.shortcut(x)
        out = x.permute(position_coding("NCWH", "NHCW"))
        out = self.conv1(out)
        out = self.bn1(out.permute(position_coding("NHCW", "NCWH")))
        out2 = x.permute(position_coding("NCWH", "NWHC"))
        out2 = self.conv2(out2)
        out2 = out2.permute(position_coding("NWHC", "NCWH"))
        out2 = self.bn2(out2)
        out = torch.cat([out, out2], 1)
        out = F.gelu(self.conv4(out))
        out = F.gelu(self.bn3(self.conv3(out)) + self.shortcut(x))
        return out

class shuffle_block2(nn.Module):
    def __init__(self, in_ch1, out_ch1, bn_ch) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch1, out_ch1, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_ch1, out_ch1, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(bn_ch, bn_ch, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(bn_ch, bn_ch, 3, 1, 1, groups=bn_ch)
        # self.relu1 = relu1
        # self.relu2 = relu2
        # self.linear = nn.Conv2d(bn_ch, bn_ch, kernel_size=1, stride=1, padding=0)
        # if norm1:
        # self.bn1 = nn.BatchNorm2d(bn_ch)
        # self.bn2 = nn.BatchNorm2d(bn_ch)
        self.bn3 = nn.BatchNorm2d(bn_ch)
        self.bn1 = nn.Sequential()
        self.bn2 = nn.Sequential()
        # else:
        #     self.bn1 = nn.Sequential()
        # self.pos_code1 = pos_code1
        # self.pos_code2 = pos_code2
        
        # if norm2:
        #     self.bn2 = nn.BatchNorm2d(bn_ch)
        # else:
        #     self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
    
    def forward(self, x):
        # shortcut = self.shortcut(x)
        out = x.permute(position_coding("NCWH", "NHCW"))
        out = self.conv1(out)
        out = self.bn1(out.permute(position_coding("NHCW", "NCWH")))
        out2 = x.permute(position_coding("NCWH", "NWHC"))
        out2 = self.conv2(out2)
        out2 = out2.permute(position_coding("NWHC", "NCWH"))
        out2 = self.bn2(out2)
        out = out + out2
        out = F.gelu(self.conv3(out))
        out = F.gelu(self.bn3(self.conv4(out)) + self.shortcut(x))
        return out

class shuffle_plug_3(nn.Module):
    def __init__(self, in_ch1, out_ch1, bn_ch) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch1, out_ch1, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_ch1, out_ch1, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(2 * bn_ch, bn_ch, kernel_size=1, stride=1, padding=0)
        # self.relu1 = relu1
        # self.relu2 = relu2
        # self.linear = nn.Conv2d(bn_ch, bn_ch, kernel_size=1, stride=1, padding=0)
        # if norm1:
        # self.bn1 = nn.BatchNorm2d(bn_ch)
        # self.bn2 = nn.BatchNorm2d(bn_ch)
        self.bn3 = nn.BatchNorm2d(bn_ch)
        self.bn1 = nn.Sequential()
        self.bn2 = nn.Sequential()
        # else:
        #     self.bn1 = nn.Sequential()
        # self.pos_code1 = pos_code1
        # self.pos_code2 = pos_code2
        
        # if norm2:
        #     self.bn2 = nn.BatchNorm2d(bn_ch)
        # else:
        #     self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
    
    def forward(self, x):
        # shortcut = self.shortcut(x)
        out = x.permute(position_coding("NCWH", "NHCW"))
        out = self.bn1(out.permute(position_coding("NHCW", "NCWH")))
        out2 = x.permute(position_coding("NCWH", "NWHC"))
        out2 = self.conv2(out2)
        out2 = out2.permute(position_coding("NWHC", "NCWH"))
        out2 = self.bn2(out2)
        out = torch.cat([out, out2], 1)
        out = F.gelu(self.bn3(self.conv3(out)) + self.shortcut(x))
        return out

class shuffle_plug_4(nn.Module):
    def __init__(self, in_ch1, out_ch1, bn_ch) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch1, out_ch1, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_ch1, out_ch1, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(bn_ch, bn_ch, kernel_size=3, stride=1, padding=1, groups=bn_ch)
        # self.relu1 = relu1
        # self.relu2 = relu2
        # self.linear = nn.Conv2d(bn_ch, bn_ch, kernel_size=1, stride=1, padding=0)
        # if norm1:
        # self.bn1 = nn.BatchNorm2d(bn_ch)
        # self.bn2 = nn.BatchNorm2d(bn_ch)
        self.bn3 = nn.BatchNorm2d(bn_ch)
        self.bn1 = nn.BatchNorm2d(bn_ch)
        self.bn2 = nn.BatchNorm2d(bn_ch)
        # else:
        #     self.bn1 = nn.Sequential()
        # self.pos_code1 = pos_code1
        # self.pos_code2 = pos_code2
        
        # if norm2:
        #     self.bn2 = nn.BatchNorm2d(bn_ch)
        # else:
        #     self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
    
    def forward(self, x):
        # shortcut = self.shortcut(x)
        out = x.permute(position_coding("NCWH", "NHCW"))
        out = self.conv1(out)
        out = self.bn1(out.permute(position_coding("NHCW", "NCWH")))
        out2 = x.permute(position_coding("NCWH", "NWHC"))
        out2 = self.conv2(out2)
        out2 = out2.permute(position_coding("NWHC", "NCWH"))
        out2 = self.bn2(out2)
        out = out + out2
        out = F.gelu(self.bn3(self.conv3(out)) + self.shortcut(x))
        return out

class adjustable_plug(nn.Module):
    def __init__(self, plug, feature_size, ch, cycle_size=(1, 5), cycle_size2=(5, 1)) -> None:
        super().__init__()
        self.plug = plug
        self.shortcut = nn.Sequential()
        if plug == 0 or plug == 1:
            self.conv = nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=1, padding=0)
        elif plug == 2:
            self.conv = CycleFC(feature_size, feature_size, cycle_size, 1, (0, 0))
        elif plug == 3:
            self.conv = CycleFC(feature_size, feature_size, cycle_size2, 1, (0, 0))
        
        self.bn = nn.BatchNorm2d(ch)
    
    def forward(self, x):
        if self.plug == 0 or self.plug == 2:
            out = x.permute(position_coding("NCWH", "NHCW"))
            out = self.conv(out)
            out = self.bn(out.permute(position_coding("NHCW", "NCWH")))
        elif self.plug == 1 or self.plug == 3:
            out = x.permute(position_coding("NCWH", "NWHC"))
            out = self.conv(out)
            out = self.bn(out.permute(position_coding("NWHC", "NCWH")))

        out = F.gelu(out + self.shortcut(x))

        return out

class shuffle_plug_with_cycle(nn.Module):
    def __init__(self, feature_size, bn_ch, cycle_size=(1, 5), cycle_size2=(5, 1)) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.cycle_conv1 = CycleFC(feature_size, feature_size, cycle_size, 1, (0, 0))
        self.cycle_conv2 = CycleFC(feature_size, feature_size, cycle_size2, 1, (0, 0))
        # self.conv3 = nn.Conv2d(bn_ch, bn_ch, kernel_size=3, stride=1, padding=1, groups=bn_ch)
        # self.relu1 = relu1
        # self.relu2 = relu2
        # self.linear = nn.Conv2d(bn_ch, bn_ch, kernel_size=1, stride=1, padding=0)
        # if norm1:
        # self.bn1 = nn.BatchNorm2d(bn_ch)
        # self.bn2 = nn.BatchNorm2d(bn_ch)
        # self.bn3 = nn.BatchNorm2d(bn_ch)
        self.bn1 = nn.BatchNorm2d(bn_ch)
        self.bn2 = nn.BatchNorm2d(bn_ch)
        self.bn_cycle1 = nn.BatchNorm2d(bn_ch)
        self.bn_cycle2 = nn.BatchNorm2d(bn_ch)

        self.shortcut = nn.Sequential()
    
    def forward(self, x):
        # shortcut = self.shortcut(x)
        out = x.permute(position_coding("NCWH", "NHCW"))
        out_cycle1 = self.cycle_conv1(out)
        out = self.conv1(out)
        out = self.bn1(out.permute(position_coding("NHCW", "NCWH")))
        out_cycle1 = self.bn_cycle1(out_cycle1.permute(position_coding("NHCW", "NCWH")))
        out2 = x.permute(position_coding("NCWH", "NWHC"))
        out_cycle2 = self.cycle_conv2(out2)
        out2 = self.conv2(out2)
        out2 = out2.permute(position_coding("NWHC", "NCWH"))
        out_cycle2 = self.bn_cycle2(out_cycle2.permute(position_coding("NWHC", "NCWH")))
        out2 = self.bn2(out2)
        out = out + out2 + out_cycle1 + out_cycle2
        # out = F.gelu(self.bn3(self.conv3(out)) + self.shortcut(x))
        out = F.gelu(out + self.shortcut(x))
        return out

class shuffle_plug_padding(nn.Module):
    def __init__(self, in_ch1, out_ch1, bn_ch, kernel_size=7) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch1, out_ch1 + 2 * (kernel_size // 2), kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_ch1, out_ch1 + 2 * (kernel_size // 2), kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(bn_ch, bn_ch, kernel_size=kernel_size, stride=1, padding=0, groups=bn_ch)
        # self.relu1 = relu1
        # self.relu2 = relu2
        # self.linear = nn.Conv2d(bn_ch, bn_ch, kernel_size=1, stride=1, padding=0)
        # if norm1:
        # self.bn1 = nn.BatchNorm2d(bn_ch)
        # self.bn2 = nn.BatchNorm2d(bn_ch)
        self.bn3 = nn.BatchNorm2d(bn_ch)
        self.bn1 = nn.BatchNorm2d(bn_ch)
        self.bn2 = nn.BatchNorm2d(bn_ch)
        # else:
        #     self.bn1 = nn.Sequential()
        # self.pos_code1 = pos_code1
        # self.pos_code2 = pos_code2
        
        # if norm2:
        #     self.bn2 = nn.BatchNorm2d(bn_ch)
        # else:
        #     self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
    
    def forward(self, x):
        # shortcut = self.shortcut(x)
        out = x.permute(position_coding("NCWH", "NHCW"))
        out = self.conv1(out)
        out = self.bn1(out.permute(position_coding("NHCW", "NCWH")))
        out = out.permute(position_coding("NCWH", "NWHC"))
        out = self.conv2(out)
        out = out.permute(position_coding("NWHC", "NCWH"))
        out = self.bn2(out)
        out = self.bn3(self.conv3(out))
        out = F.gelu(out + self.shortcut(x))
        return out

class shuffle_plug_padding2(nn.Module):
    def __init__(self, in_ch1, out_ch1, bn_ch, kernel_size=9) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch1, out_ch1 + 2 * (kernel_size // 2), kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_ch1, out_ch1 + 2 * (kernel_size // 2), kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(bn_ch, bn_ch, kernel_size=kernel_size, stride=1, padding=0, groups=bn_ch)
        self.pad2 = (kernel_size // 2, kernel_size // 2)
        self.pad1 = (0, 0, kernel_size // 2, kernel_size // 2)
        # self.relu1 = relu1
        # self.relu2 = relu2
        # self.linear = nn.Conv2d(bn_ch, bn_ch, kernel_size=1, stride=1, padding=0)
        # if norm1:
        # self.bn1 = nn.BatchNorm2d(bn_ch)
        # self.bn2 = nn.BatchNorm2d(bn_ch)
        self.bn3 = nn.BatchNorm2d(bn_ch)
        self.bn1 = nn.BatchNorm2d(bn_ch)
        self.bn2 = nn.BatchNorm2d(bn_ch)
        # else:
        #     self.bn1 = nn.Sequential()
        # self.pos_code1 = pos_code1
        # self.pos_code2 = pos_code2
        
        # if norm2:
        #     self.bn2 = nn.BatchNorm2d(bn_ch)
        # else:
        #     self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
    
    def forward(self, x):
        # shortcut = self.shortcut(x)
        out = x.permute(position_coding("NCWH", "NHCW"))
        out = self.conv1(out)
        out = self.bn1(out.permute(position_coding("NHCW", "NCWH")))
        out = F.pad(out, self.pad1, mode='constant', value=0)
        out2 = x.permute(position_coding("NCWH", "NWHC"))
        out2 = self.conv2(out2)
        out2 = out2.permute(position_coding("NWHC", "NCWH"))
        out2 = self.bn2(out2)
        out2 = F.pad(out2, self.pad2, mode='constant', value=0)
        out = out + out2
        out = F.gelu(self.bn3(self.conv3(out)) + self.shortcut(x))
        return out


class shuffle_plug_padding3(nn.Module):
    def __init__(self, in_ch1, out_ch1, bn_ch, kernel_size=9) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch1, out_ch1 + 2 * (kernel_size // 2), kernel_size=1, stride=1, padding=0)
        self.pad_conv1 = nn.Conv2d(in_ch1, 2 * (kernel_size // 2), kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_ch1, out_ch1 + 2 * (kernel_size // 2), kernel_size=1, stride=1, padding=0)
        self.pad_conv2 = nn.Conv2d(in_ch1, 2 * (kernel_size // 2), kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(bn_ch, bn_ch, kernel_size=kernel_size, stride=1, padding=0, groups=bn_ch)
        # self.pad2 = (kernel_size // 2, kernel_size // 2)
        # self.pad1 = (0, 0, kernel_size // 2, kernel_size // 2)
        # self.relu1 = relu1
        # self.relu2 = relu2
        # self.linear = nn.Conv2d(bn_ch, bn_ch, kernel_size=1, stride=1, padding=0)
        # if norm1:
        # self.bn1 = nn.BatchNorm2d(bn_ch)
        # self.bn2 = nn.BatchNorm2d(bn_ch)
        self.bn3 = nn.BatchNorm2d(bn_ch)
        self.bn1 = nn.BatchNorm2d(bn_ch)
        self.bn2 = nn.BatchNorm2d(bn_ch)
        # else:
        #     self.bn1 = nn.Sequential()
        # self.pos_code1 = pos_code1
        # self.pos_code2 = pos_code2
        
        # if norm2:
        #     self.bn2 = nn.BatchNorm2d(bn_ch)
        # else:
        #     self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
    
    def forward(self, x):
        # shortcut = self.shortcut(x)
        out = x.permute(position_coding("NCWH", "NHCW"))
        out = self.conv1(out)
        out = out.permute(position_coding("NHCW", "NCWH"))
        out_padding = out.permute(position_coding("NCWH", "NWHC"))
        out_padding = self.pad_conv1(out_padding)
        out_padding = out_padding.permute(position_coding("NWHC", "NCWH"))
        out = self.bn1(torch.cat((out, out_padding), 2))
        # out = F.pad(out, self.pad1, mode='constant', value=0)
        out2 = x.permute(position_coding("NCWH", "NWHC"))
        out2 = self.conv2(out2)
        out2 = out2.permute(position_coding("NWHC", "NCWH"))
        out_padding = out2.permute(position_coding("NCWH", "NHCW"))
        out_padding = self.pad_conv2(out_padding)
        out_padding = out_padding.permute(position_coding("NHCW", "NCWH"))
        out2 = self.bn2(torch.cat((out2, out_padding), 3))
        # out2 = F.pad(out2, self.pad2, mode='constant', value=0)
        out = out + out2
        out = F.gelu(self.bn3(self.conv3(out)) + self.shortcut(x))
        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class shuffle_plug_5(nn.Module):
    def __init__(self, in_ch1, out_ch1, bn_ch) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch1, out_ch1, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_ch1, out_ch1, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(bn_ch, bn_ch, kernel_size=3, stride=1, padding=1, groups=bn_ch)
        # self.relu1 = relu1
        # self.relu2 = relu2
        # self.linear = nn.Conv2d(bn_ch, bn_ch, kernel_size=1, stride=1, padding=0)
        # if norm1:
        # self.bn1 = nn.BatchNorm2d(bn_ch)
        # self.bn2 = nn.BatchNorm2d(bn_ch)
        self.bn3 = nn.BatchNorm2d(bn_ch // 2)
        self.bn1 = nn.BatchNorm2d(bn_ch // 2)
        self.bn2 = nn.BatchNorm2d(bn_ch // 2)
        # else:
        #     self.bn1 = nn.Sequential()
        # self.pos_code1 = pos_code1
        # self.pos_code2 = pos_code2
        
        # if norm2:
        #     self.bn2 = nn.BatchNorm2d(bn_ch)
        # else:
        #     self.bn2 = nn.Sequential()

        self.shortcut = nn.Conv2d(bn_ch, bn_ch // 2, 1, 1, 0)
    
    def forward(self, x):
        shortcut = self.bn3(self.shortcut(x))
        # out = self.shortcut(x)
        out = shortcut.permute(position_coding("NCWH", "NHCW"))
        out = self.conv1(out)
        out = self.bn1(out.permute(position_coding("NHCW", "NCWH")))
        out2 = shortcut.permute(position_coding("NCWH", "NWHC"))
        out2 = self.conv2(out2)
        out2 = out2.permute(position_coding("NWHC", "NCWH"))
        out2 = self.bn2(out2)
        out = out + out2
        out = torch.cat([shortcut, F.gelu(out)], 1)
        # out = torch.cat([out, shortcut], 1)
        # out = F.gelu(self.bn3(torch.cat([out, shortcut], 1)))
        return out

def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=kernel_size // 2),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )


# class convMixer_block_v1(nn.Module):
#     def __init__(self, feature_size, ch):
#         super().__init__()
#         self.conv1 = nn.Conv2d(feature_size, feature_size, 1, 1)
#         self.conv2 = nn.Conv2d(feature_size, feature_size, 1, 1)
#         self.conv3 = nn.Conv2d(ch, ch, 1, 1)

#     def forward(self, x):


class shuffle_plug_7(nn.Module):
    def __init__(self, in_ch1, out_ch1, bn_ch) -> None:
        super().__init__()
        self.conv1 = SoftMaxPointwiseConv(in_ch1, out_ch1)
        self.conv2 = SoftMaxPointwiseConv(in_ch1, out_ch1)
        self.conv3 = nn.Conv2d(bn_ch, bn_ch, kernel_size=3, stride=1, padding=1, groups=bn_ch)
        # self.relu1 = relu1
        # self.relu2 = relu2
        # self.linear = nn.Conv2d(bn_ch, bn_ch, kernel_size=1, stride=1, padding=0)
        # if norm1:
        # self.bn1 = nn.BatchNorm2d(bn_ch)
        # self.bn2 = nn.BatchNorm2d(bn_ch)
        self.bn3 = nn.BatchNorm2d(bn_ch)
        self.bn1 = nn.BatchNorm2d(bn_ch)
        self.bn2 = nn.BatchNorm2d(bn_ch)
        # else:
        #     self.bn1 = nn.Sequential()
        # self.pos_code1 = pos_code1
        # self.pos_code2 = pos_code2
        
        # if norm2:
        #     self.bn2 = nn.BatchNorm2d(bn_ch)
        # else:
        #     self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
    
    def forward(self, x):
        # shortcut = self.shortcut(x)
        out = x.permute(position_coding("NCWH", "NHCW"))
        out = self.conv1(out)
        out = self.bn1(out.permute(position_coding("NHCW", "NCWH")))
        out2 = x.permute(position_coding("NCWH", "NWHC"))
        out2 = self.conv2(out2)
        out2 = out2.permute(position_coding("NWHC", "NCWH"))
        out2 = self.bn2(out2)
        out = out + out2
        out = F.gelu(self.bn3(self.conv3(out)))
        return out

class shuffle_plug_6(nn.Module):
    def __init__(self, in_ch1, out_ch1, bn_ch) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch1, out_ch1, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_ch1, out_ch1, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(2 * bn_ch, bn_ch, kernel_size=3, stride=1, padding=1, groups=bn_ch)
        # self.relu1 = relu1
        # self.relu2 = relu2
        # self.linear = nn.Conv2d(bn_ch, bn_ch, kernel_size=1, stride=1, padding=0)
        # if norm1:
        # self.bn1 = nn.BatchNorm2d(bn_ch)
        # self.bn2 = nn.BatchNorm2d(bn_ch)
        self.bn3 = nn.BatchNorm2d(bn_ch)
        self.bn1 = nn.BatchNorm2d(bn_ch)
        self.bn2 = nn.BatchNorm2d(bn_ch)
        self.bn_ch = bn_ch
        self.in_ch = in_ch1
        # else:
        #     self.bn1 = nn.Sequential()
        # self.pos_code1 = pos_code1
        # self.pos_code2 = pos_code2
        
        # if norm2:
        #     self.bn2 = nn.BatchNorm2d(bn_ch)
        # else:
        #     self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
    
    def forward(self, x):
        # shortcut = self.shortcut(x)
        out = x.permute(position_coding("NCWH", "NHCW"))
        out = self.conv1(out)
        out = self.bn1(out.permute(position_coding("NHCW", "NCWH")))
        out2 = x.permute(position_coding("NCWH", "NWHC"))
        out2 = self.conv2(out2)
        out2 = out2.permute(position_coding("NWHC", "NCWH"))
        out2 = self.bn2(out2)
        out = torch.cat([out, out2], 1)
        out = out.reshape(-1, self.bn_ch, 2, self.in_ch, self.in_ch)
        out = out.transpose(1, 2)
        out = out.reshape(-1, 2 * self.bn_ch, self.in_ch, self.in_ch)
        out = F.gelu(self.bn3(self.conv3(out)) + self.shortcut(x))
        return out


class TestBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, feature_size, stride=1, shuffle=False):
        super(TestBlock, self).__init__()
        if shuffle:
            self.conv_w1 = nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv_h1 = nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv_w2 = nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv_h2 = nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv_merge1 = nn.Conv2d(in_planes * 2, in_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv_merge2 = nn.Conv2d(in_planes * 2, in_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn_merge = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shuffle = shuffle

        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != planes:
        #     self.shortcut = nn.Sequential(
        #         nn.AvgPool2d((2, 2)),
        #         nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
        #         nn.BatchNorm2d(self.expansion * planes)
        #     )

    def forward(self, x):
        if self.shuffle:
            out = x.permute(position_coding("NCWH", "NHCW"))
            out = self.conv_w1(out)
            out = out.permute(position_coding("NHCW", "NCWH"))
            out2 = x.permute(position_coding("NCWH", "NWHC"))
            out2 = self.conv_h1(out2)
            out2 = out2.permute(position_coding("NWHC", "NCWH"))
            out = torch.cat([out, out2], 1)
            out = self.conv_merge1(out)
            out2 = out.permute(position_coding("NCWH", "NHCW"))
            out2 = self.conv_w2(out2)
            out2 = out2.permute(position_coding("NHCW", "NCWH"))
            out = out.permute(position_coding("NCWH", "NWHC"))
            out = self.conv_h2(out)
            out = out.permute(position_coding("NWHC", "NCWH"))
            out = torch.cat([out, out2], 1)
            out = self.bn_merge(self.conv_merge2(out))
            out = F.gelu(self.bn1(self.conv1(out)))
            out = F.gelu(self.bn2(self.conv2(out)))
        else:
            out = F.gelu(self.bn1(self.conv1(x)))
            out = F.gelu(self.bn2(self.conv2(out)))
        return out

class CycleTwist(nn.Module):
    def __init__(self, planes, feature_size, kernel_size=(1, 5)):
        super().__init__()
        self.cycle_conv1 = CycleFC(feature_size, feature_size, kernel_size, 1, (0, 0))
        self.cycle_conv2 = CycleFC(feature_size, feature_size, kernel_size, 1, (0, 0))
        self.conv3 = nn.Conv2d(2 * planes, planes, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(planes)
        self.bn1 = nn.Sequential()
        self.bn2 = nn.Sequential()
        self.shortcut = nn.Sequential()

    def forward(self, x):
        # shortcut = self.shortcut(x)
        out = x.permute(position_coding("NCWH", "NHCW"))
        out = self.cycle_conv1(out)
        out = self.bn1(out.permute(position_coding("NHCW", "NCWH")))
        out2 = x.permute(position_coding("NCWH", "NWHC"))
        out2 = self.cycle_conv2(out2)
        out2 = out2.permute(position_coding("NWHC", "NCWH"))
        out2 = self.bn2(out2)
        out = torch.cat([out, out2], 1)
        out = F.gelu(self.bn3(self.conv3(out)) + self.shortcut(x))
        # out = self.shortcut(x)
        return out

class CycleTwist2(nn.Module):
    def __init__(self, planes, feature_size, kernel_size1=(1, 5), kernel_size2=(5, 1)):
        super().__init__()
        self.cycle_conv1 = CycleFC(feature_size, feature_size, kernel_size1, 1, (0, 0))
        self.cycle_conv2 = CycleFC(feature_size, feature_size, kernel_size2, 1, (0, 0))
        self.conv3 = nn.Conv2d(2 * planes, planes, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(planes)
        self.bn1 = nn.Sequential()
        self.bn2 = nn.Sequential()
        self.shortcut = nn.Sequential()

    def forward(self, x):
        # shortcut = self.shortcut(x)
        out = x.permute(position_coding("NCWH", "NHCW"))
        out = self.cycle_conv1(out)
        out = self.bn1(out.permute(position_coding("NHCW", "NCWH")))
        out2 = x.permute(position_coding("NCWH", "NWHC"))
        out2 = self.cycle_conv2(out2)
        out2 = out2.permute(position_coding("NWHC", "NCWH"))
        out2 = self.bn2(out2)
        out = torch.cat([out, out2], 1)
        out = F.gelu(self.bn3(self.conv3(out)) + self.shortcut(x))
        # out = F.gelu(self.bn3(self.conv3(out)))
        # out = self.shortcut(x)
        return out

class SoftMaxPointwiseConv(nn.Module):
    def __init__(self, in_ch, out_ch, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_ch))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        self.weight.data = nn.Softmax(dim=1)(self.weight.data)
        return F.conv2d(x, self.weight, self.bias)

class SoftMaxConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding='same', bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, kernel_size, kernel_size))
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size= kernel_size
        self.padding = padding
        self.stride = stride
    
        if bias:
            self.bias = nn.Parameter(torch.empty(out_ch))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        weight_data = self.weight.data.view(self.out_ch, self.in_ch, self.kernel_size * self.kernel_size)
        att = nn.Softmax(dim=2)(weight_data / weight_data.min(dim=-1).values.unsqueeze(-1)).view(self.out_ch, self.in_ch, self.kernel_size, self.kernel_size)
        return F.conv2d(x, self.weight * att, self.bias, self.stride, padding=self.padding)

class CycleFC(nn.Module):
    """
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,  # re-defined kernel_size, represent the spatial area of staircase FC
        stride: int = 1,
        padding: tuple = (0, 0),
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(CycleFC, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        # if padding != 0:
        #     raise ValueError('padding must be 0')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = padding
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))  # kernel size == 1

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('offset', self.gen_offset())

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def gen_offset(self):
        """
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
            out_height, out_width]): offsets to be applied for each position in the
            convolution kernel.
        """
        offset = torch.empty(1, self.in_channels*2, 1, 1)
        start_idx = (self.kernel_size[0] * self.kernel_size[1]) // 2
        assert self.kernel_size[0] == 1 or self.kernel_size[1] == 1, self.kernel_size
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                offset[0, 2 * i + 1, 0, 0] = (i + start_idx) % self.kernel_size[1] - (self.kernel_size[1] // 2)
            else:
                offset[0, 2 * i + 0, 0, 0] = (i + start_idx) % self.kernel_size[0] - (self.kernel_size[0] // 2)
                offset[0, 2 * i + 1, 0, 0] = 0
        return offset

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        """
        B, C, H, W = input.size()
        return deform_conv2d_tv(input, self.offset.expand(B, -1, H, W), self.weight, self.bias, stride=self.stride,
                                padding=self.padding, dilation=self.dilation)

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)
    