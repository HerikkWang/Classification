# from msilib.schema import Shortcut
from turtle import forward
from numpy import block, pad
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable
from torch.nn.modules.conv import Conv2d
from torchsummary import summary

class CubeNormLayer(nn.Module):
    def __init__(self, eps=1e-5) -> None:
        super(CubeNormLayer, self).__init__()
        self.eps = eps
    
    def forward(self, x):
        cube_mean = torch.mean(x, dim=(1, 2, 3))
        cube_var = torch.var(x, dim=(1, 2, 3))
        res = (x - cube_mean[:, None, None, None]) / torch.sqrt(cube_var[:, None, None, None] + self.eps)
        return res


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

def position_coding(pos1, pos2):
    char_dict = {}
    for i in range(len(pos1)):
        char_dict[pos1[i]] = i
    
    code = []
    for i in range(len(pos2)):
        code.append(char_dict[pos2[i]])
    
    return tuple(code)

class cube_block(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=1, padding=0, norm="BatchNorm", block_norm="BatchNorm", twist=True) -> None:
        super(cube_block, self).__init__()
        self.norm = norm
        self.twist = twist
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        if norm == "BatchNorm":
            self.bn1 = nn.BatchNorm2d(out_planes)
            self.bn2 = nn.BatchNorm2d(out_planes)
            self.bn3 = nn.BatchNorm2d(out_planes)
        elif norm == "LayerNorm":
            self.bn1 = nn.LayerNorm([out_planes, out_planes, out_planes])
            self.bn2 = nn.LayerNorm([out_planes, out_planes, out_planes])
            self.bn3 = nn.LayerNorm([out_planes, out_planes, out_planes])
        elif norm == "CubeNorm":
            self.bn1 = CubeNormLayer()
            self.bn2 = CubeNormLayer()
            self.bn3 = CubeNormLayer()
        else:
            pass
        if block_norm == "BatchNorm":
            self.block_norm = nn.BatchNorm2d(out_planes)
        elif block_norm == "LayerNorm":
            self.block_norm = nn.LayerNorm([out_planes, out_planes, out_planes])
        else:
            self.block_norm = nn.Sequential() 
        self.shortcut = nn.Sequential()
    
    def forward(self, x):
        shortcut = self.shortcut(x)
        if self.norm != "NoNorm":
            out = F.relu(self.bn1(self.conv1(x)))
            if self.twist:
                out = out.permute(0, 3, 1, 2)
            out = F.relu(self.bn2(self.conv2(out)))
            if self.twist:
                out = out.permute(0, 3, 1, 2)
            out = self.bn3(self.conv3(out))
            if self.twist:
                out = out.permute(0, 3, 1, 2)
            out += shortcut
            out = self.block_norm(F.relu(out))
        else:
            out = F.relu(self.conv1(x))
            if self.twist:
                out = out.permute(0, 3, 1, 2)
            out = F.relu(self.conv2(out))
            if self.twist:
                out = out.permute(0, 3, 1, 2)
            out = self.conv3(out)
            if self.twist:
                out = out.permute(0, 3, 1, 2)
            out += shortcut
            out = self.block_norm(F.relu(out))
        return out

class cube_block_bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=1, padding=0, norm="BatchNorm", block_norm="BatchNorm", twist=True) -> None:
        super(cube_block, self).__init__()
        self.norm = norm
        self.twist = twist
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        if norm == "BatchNorm":
            self.bn1 = nn.BatchNorm2d(out_planes)
            self.bn2 = nn.BatchNorm2d(out_planes)
            self.bn3 = nn.BatchNorm2d(out_planes)
        elif norm == "LayerNorm":
            self.bn1 = nn.LayerNorm([out_planes, out_planes, out_planes])
            self.bn2 = nn.LayerNorm([out_planes, out_planes, out_planes])
            self.bn3 = nn.LayerNorm([out_planes, out_planes, out_planes])
        elif norm == "CubeNorm":
            self.bn1 = CubeNormLayer()
            self.bn2 = CubeNormLayer()
            self.bn3 = CubeNormLayer()
        else:
            pass
        if block_norm == "BatchNorm":
            self.block_norm = nn.BatchNorm2d(out_planes)
        elif block_norm == "LayerNorm":
            self.block_norm = nn.LayerNorm([out_planes, out_planes, out_planes])
        else:
            self.block_norm = nn.Sequential() 
        self.shortcut = nn.Sequential()
    
    def forward(self, x):
        shortcut = self.shortcut(x)
        if self.norm != "NoNorm":
            out = F.relu(self.bn1(self.conv1(x)))
            if self.twist:
                out = out.permute(0, 3, 1, 2)
            out = F.relu(self.bn2(self.conv2(out)))
            if self.twist:
                out = out.permute(0, 3, 1, 2)
            out = self.bn3(self.conv3(out))
            if self.twist:
                out = out.permute(0, 3, 1, 2)
            out += shortcut
            out = self.block_norm(F.relu(out))
        else:
            out = F.relu(self.conv1(x))
            if self.twist:
                out = out.permute(0, 3, 1, 2)
            out = F.relu(self.conv2(out))
            if self.twist:
                out = out.permute(0, 3, 1, 2)
            out = self.conv3(out)
            if self.twist:
                out = out.permute(0, 3, 1, 2)
            out += shortcut
            out = self.block_norm(F.relu(out))
        return out


class cube_pair_block(nn.Module):
    def __init__(self, planes, dilate_planes, stride=1, kernel_size=1, padding=0, norm="BatchNorm") -> None:
        super(cube_pair_block, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(planes, dilate_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(planes, dilate_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv3 = nn.Conv2d(planes, dilate_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv4 = nn.Conv2d(dilate_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv5 = nn.Conv2d(dilate_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv6 = nn.Conv2d(dilate_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        if norm == "BatchNorm":
            self.bn1 = nn.BatchNorm2d(dilate_planes)
            self.bn2 = nn.BatchNorm2d(dilate_planes)
            self.bn3 = nn.BatchNorm2d(dilate_planes)
            self.bn4 = nn.BatchNorm2d(planes)
            self.bn5 = nn.BatchNorm2d(planes)
            self.bn6 = nn.BatchNorm2d(planes)
        elif norm == "LayerNorm":
            self.bn1 = nn.LayerNorm([dilate_planes, planes, planes])
            self.bn2 = nn.LayerNorm([dilate_planes, dilate_planes, planes])
            self.bn3 = nn.LayerNorm([dilate_planes, dilate_planes, dilate_planes])
            self.bn4 = nn.LayerNorm([planes, dilate_planes, dilate_planes])
            self.bn5 = nn.LayerNorm([planes, planes, dilate_planes])
            self.bn6 = nn.LayerNorm([planes, planes, planes])
        else:
            pass
        self.shortcut = nn.Sequential()
    
    def forward(self, x):
        shortcut = self.shortcut(x)
        if self.norm != "NoNorm":
            out = F.relu(self.bn1(self.conv1(x)))
            out = out.permute(0, 3, 1, 2)
            out = F.relu(self.bn2(self.conv2(out)))
            out = out.permute(0, 3, 1, 2)
            out = F.relu(self.bn3(self.conv3(out)))
            out = out.permute(0, 3, 1, 2)
            out = F.relu(self.bn4(self.conv4(out)))
            out = out.permute(0, 3, 1, 2)
            out = F.relu(self.bn5(self.conv5(out)))
            out = out.permute(0, 3, 1, 2)
            out = self.bn6(self.conv6(out))
            out = out.permute(0, 3, 1, 2)
            out += shortcut
            out = F.relu(out)
        else:
            out = F.relu(self.conv1(x))
            out = out.permute(0, 3, 1, 2)
            out = F.relu(self.conv2(out))
            out = out.permute(0, 3, 1, 2)
            out = F.relu(self.conv3(out))
            out = out.permute(0, 3, 1, 2)
            out = F.relu(self.conv4(out))
            out = out.permute(0, 3, 1, 2)
            out = F.relu(self.conv5(out))
            out = out.permute(0, 3, 1, 2)
            out = self.conv6(out)
            out = out.permute(0, 3, 1, 2)
            out += shortcut
            out = F.relu(out)
        return out

class cube_downsample(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, norm='LayerNorm') -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = norm
        if norm == "LayerNorm":
            self.norm_layer = nn.LayerNorm([out_planes, out_planes, out_planes])
        elif norm == "BatchNorm":
            self.norm_layer = nn.BatchNorm2d(out_planes)
        elif norm == "CubeNorm":
            self.norm_layer = CubeNormLayer()
        else:
            pass
        # if block_norm == "BatchNorm":
        #     self.block_norm = nn.BatchNorm2d(out_planes)
        # elif block_norm == "LayerNorm":
        #     self.block_norm == nn.LayerNorm([out_planes, out_planes, out_planes])
        # else:
        #     pass
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out.permute(0, 3, 1, 2))
        out = self.conv2(out)
        out = F.relu(out.permute(0, 3, 1, 2))
        out = self.conv3(out)
        out = F.relu(out.permute(0, 3, 1, 2))
        if self.norm != "NoNorm":
            out = self.norm_layer(out)
        
        return out

class cube_downsample_v2(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, norm='LayerNorm') -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=2, stride=2, padding=0),
            CubeNormLayer(),
        )       
        self.norm = norm
        if norm == "LayerNorm":
            self.norm_layer = nn.LayerNorm([out_planes, out_planes, out_planes])
        elif norm == "BatchNorm":
            self.norm_layer = nn.BatchNorm2d(out_planes)
        elif norm == "CubeNorm":
            self.norm_layer = CubeNormLayer()
        else:
            pass
        # if block_norm == "BatchNorm":
        #     self.block_norm = nn.BatchNorm2d(out_planes)
        # elif block_norm == "LayerNorm":
        #     self.block_norm == nn.LayerNorm([out_planes, out_planes, out_planes])
        # else:
        #     pass
    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.norm_layer(self.conv1(x))
        out = F.relu(out.permute(0, 3, 1, 2))
        # print(out.size())
        out = self.norm_layer(self.conv2(out))
        out = F.relu(out.permute(0, 3, 1, 2))
        out = self.conv3(out)
        out = F.relu(out.permute(0, 3, 1, 2) + shortcut)
        # if self.norm != "NoNorm":
        #     out = self.norm_layer(out)
        
        return out


class cube_bottleneck(nn.Module):
    def __init__(self, plane, neck_plane, permute_code=None) -> None:
        super(cube_bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=plane, out_channels=neck_plane, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=neck_plane, out_channels=neck_plane, kernel_size=3, stride=1, padding=1, groups=8)
        self.conv3 = nn.Conv2d(in_channels=neck_plane, out_channels=plane, kernel_size=1, stride=1, padding=0)
        self.shortcut = nn.Sequential()
        self.norm_layer = CubeNormLayer()
        if permute_code:
            self.permute_code = permute_code
    
    def forward(self, x):
        out = torch.permute(x, self.permute_code)
        out = F.relu(self.norm_layer(self.conv1(x)))
        out = F.relu(self.norm_layer(self.conv2(out)))
        out = self.norm_layer(self.conv3(out))
        out = F.relu(out + self.shortcut(x))
        return out

class Cube_net_cn(nn.Module):
    def __init__(self, cube_size=32, norm="NoNorm", num_classes=100, res_cube_size=4):
        super(Cube_net_cn, self).__init__()
        self.cube_size = cube_size
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.block1 = cube_block(norm=norm, in_planes=cube_size, out_planes=cube_size, stride=1, kernel_size=5, padding=2, block_norm="NoNorm")
        # self.block1_bn = nn.BatchNorm2d(32)
        self.block2 = cube_block(norm=norm, in_planes=cube_size, out_planes=cube_size, stride=1, kernel_size=5, padding=2, block_norm="NoNorm")
        # self.block2_bn = nn.BatchNorm2d(32)
        # self.downsample1 = cube_downsample(32, 16)
        self.block3 = cube_block(norm=norm, in_planes=cube_size, out_planes=cube_size, stride=1, kernel_size=3, padding=1, block_norm="NoNorm")
        # self.block3_bn = nn.BatchNorm2d(32)
        self.block4 = cube_block(norm=norm, in_planes=cube_size, out_planes=cube_size, stride=1, kernel_size=3, padding=1, block_norm="NoNorm")
        self.downsample2 = cube_downsample_v2(32, 16, norm="CubeNorm")
        # self.block4_bn = nn.BatchNorm2d(32)
        self.block5 = cube_block(norm=norm, in_planes=16, out_planes=16, stride=1, kernel_size=1, padding=0, block_norm="NoNorm")
        # self.block5_bn = nn.BatchNorm2d(32)
        self.block6 = cube_block(norm=norm, in_planes=16, out_planes=16, stride=1, kernel_size=1, padding=0, block_norm="NoNorm")
        self.block7 = cube_block(norm=norm, in_planes=16, out_planes=16, stride=1, kernel_size=1, padding=0, block_norm="NoNorm")
        self.block8 = cube_block(norm=norm, in_planes=16, out_planes=16, stride=1, kernel_size=1, padding=0, block_norm="NoNorm")
        self.block10 = cube_block(norm=norm, in_planes=16, out_planes=16, stride=1, kernel_size=1, padding=0, block_norm="NoNorm")
        self.norm_layer = CubeNormLayer()
        # self.block10_bn = nn.BatchNorm2d(16)

        # self.block9_shortcut = nn.Sequential()

        self.linear = nn.Linear(res_cube_size ** 3, num_classes)
        self.apply(_weights_init)
    
    def forward(self, x):
        # out = x.reshape(-1, 3, 2, 16, 2, 16)
        # out = out.permute(0, 1, 2, 4, 3, 5)
        # out = out.reshape(-1, 12, 16, 16)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.block1(out)
        out = self.norm_layer(out)
        # out = torch.permute(out, position_coding("NCWH", "NHCW"))
        # out = self.block1_bn(out)
        # out = torch.permute(out, position_coding("NHCW", "NCWH"))
        out = self.block2(out)
        out = self.norm_layer(out)
        # out = torch.permute(out, position_coding("NCWH", "NWHC"))
        # out = self.block2_bn(out)
        # out = torch.permute(out, position_coding("NWHC", "NCWH"))
        out = self.block3(out)
        out = self.norm_layer(out)
        # out = torch.permute(out, position_coding("NCWH", "NWHC"))
        # out = self.block3_bn(out)
        # out = torch.permute(out, position_coding("NWHC", "NCWH"))
        out = self.block4(out)
        out = self.norm_layer(out)
        out = self.downsample2(out)
        # out = torch.permute(out, position_coding("NCWH", "NHCW"))
        # out = self.block4_bn(out)
        # out = torch.permute(out, position_coding("NHCW", "NCWH"))
        
        out = self.block5(out)
        out = self.norm_layer(out)
        # out = torch.permute(out, position_coding("NCWH", "NWHC"))
        # out = self.block5_bn(out)
        # out = torch.permute(out, position_coding("NWHC", "NCWH"))
        out = self.block6(out)
        out = self.norm_layer(out)
        # out = self.block6_bn(out)
        # out = self.block6_bn(out)
        # out = self.downsample3(out)
        out = self.block7(out)
        out = self.norm_layer(out)
        # out = torch.permute(out, position_coding("NCWH", "NHCW"))
        # out = self.block7_bn(out)
        # out = torch.permute(out, position_coding("NHCW", "NCWH"))
        # out = self.block7_bn(out)
        out = self.block8(out)
        out = self.norm_layer(out)
        # out = torch.permute(out, position_coding("NCWH", "NWHC"))
        # out = self.block8_bn(out)
        # out = torch.permute(out, position_coding("NWHC", "NCWH"))
        # out = self.block9(out)
        # out = self.norm_layer(out)
        # out = self.block9_bn(out)
        out = self.block10(out)
        out = self.norm_layer(out)
        # out = self.block10_bn(out)
        # out = self.block8_bn(out)
        out = out.reshape(-1, 4, 4, 4, 4, 4, 4)
        out = out.permute(0, 1, 5, 3, 4, 2, 6)
        out = out.reshape(-1, 64, 4, 4, 4)
        out = F.avg_pool3d(out, 4)
        out = out.view(-1, 64)

        # out = out.reshape(out.size(0), out.size(1)*out.size(2)*out.size(3))
        out = self.linear(out)
        return out

class Cube_net_bottleneck(nn.Module):
    def __init__(self, cube_size=32, neck_plane=64, norm="NoNorm", num_classes=100, res_cube_size=4):
        super(Cube_net_bottleneck, self).__init__()
        self.cube_size = cube_size
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.block1 = cube_bottleneck(cube_size, neck_plane, position_coding("NCWH", "NHCW"))
        self.block2 = cube_bottleneck(cube_size, neck_plane, position_coding("NHCW", "NWHC"))
        self.block3 = cube_bottleneck(cube_size, neck_plane, position_coding("NWHC", "NCWH"))
        # self.block4 = cube_bottleneck(cube_size, neck_plane, position_coding("NCWH", "NHCW"))
        # self.block5 = cube_bottleneck(cube_size, neck_plane, position_coding("NHCW", "NWHC"))
        # self.block6 = cube_bottleneck(cube_size, neck_plane, position_coding("NWHC", "NCWH"))
        self.downsample = cube_downsample_v2(in_planes=32, out_planes=16, kernel_size=1, stride=1, norm="CubeNorm")
        # self.block4_bn = nn.BatchNorm2d(32)
        self.block7 = cube_bottleneck(16, 64, position_coding("NCWH", "NHCW"))
        self.block8 = cube_bottleneck(16, 64, position_coding("NHCW", "NWHC"))
        self.block9 = cube_bottleneck(16, 64, position_coding("NWHC", "NCWH"))
        self.block10 = cube_bottleneck(16, 64, position_coding("NCWH", "NHCW"))
        self.block11 = cube_bottleneck(16, 64, position_coding("NHCW", "NWHC"))
        self.block12 = cube_bottleneck(16, 64, position_coding("NWHC", "NCWH"))

        self.norm_layer = CubeNormLayer()
        # self.block10_bn = nn.BatchNorm2d(16)

        # self.block9_shortcut = nn.Sequential()

        self.linear = nn.Linear(res_cube_size ** 3, num_classes)
        self.apply(_weights_init)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.block1(out)
        # out = self.block2(out)
        # out = self.block3(out)
        # out = self.block4(out)
        # out = self.block5(out)
        # out = self.block6(out)
        out = self.downsample(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        out = self.block10(out)
        out = self.block11(out)
        out = self.block12(out)
        out = out.reshape(-1, 4, 4, 4, 4, 4, 4)
        out = out.permute(0, 1, 5, 3, 4, 2, 6)
        out = out.reshape(-1, 64, 4, 4, 4)
        out = F.avg_pool3d(out, 4)
        out = out.view(-1, 64)
        out = self.linear(out)
        return out

class ResNet_with_cube_block(nn.Module):
    def __init__(self, num_classes=10, norm='CubeNorm'):
        super(ResNet_with_cube_block, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # block1
        self.block1_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn1 = nn.BatchNorm2d(16)
        self.block1_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn2 = nn.BatchNorm2d(16)
        self.block1_shortcut = nn.Sequential()
        # block2
        self.block2_conv1 = nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn1 = nn.BatchNorm2d(8)
        self.block2_conv2 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn2 = nn.BatchNorm2d(32)
        self.block2_shortcut = nn.Sequential()
        # block3
        self.block3_conv1 = nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn1 = nn.BatchNorm2d(8)
        self.block3_conv2 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn2 = nn.BatchNorm2d(32)
        self.block3_shortcut = nn.Sequential()
        # block4
        self.block4_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.block4_bn1 = nn.BatchNorm2d(16)
        self.block4_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block4_bn2 = nn.BatchNorm2d(16)
        self.block4_shortcut = nn.Sequential(
                            nn.Conv2d(16, 16, kernel_size=2, stride=2, bias=False),
                            nn.BatchNorm2d(16)
                        )
        self.block5 = cube_bottleneck(16, 128, position_coding("NCWH", "NHCW"))
        self.block6 = cube_bottleneck(16, 128, position_coding("NHCW", "NWHC"))
        self.block7 = cube_bottleneck(16, 128, position_coding("NWHC", "NCWH"))
        self.block8 = cube_bottleneck(16, 128, position_coding("NCWH", "NHCW"))
        self.block9 = cube_bottleneck(16, 128, position_coding("NHCW", "NWHC"))
        self.block10 = cube_bottleneck(16, 128, position_coding("NWHC", "NCWH"))
        
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        # block1
        shortcut1 = self.block1_shortcut(out)
        out = F.relu(self.block1_bn1(self.block1_conv1(out)))
        out = self.block1_bn2(self.block1_conv2(out))
        out += shortcut1
        out = F.relu(out)
        out = out.permute(0, 3, 1, 2)
        # block2
        shortcut2 = self.block2_shortcut(out)
        out = F.relu(self.block2_bn1(self.block2_conv1(out)))
        out = self.block2_bn2(self.block2_conv2(out))
        out += shortcut2
        out = F.relu(out)
        out = out.permute(0, 3, 1, 2)
        # block3
        shortcut3 = self.block3_shortcut(out)
        out = F.relu(self.block3_bn1(self.block3_conv1(out)))
        out = self.block3_bn2(self.block3_conv2(out))
        out += shortcut3
        out = F.relu(out)
        out = out.permute(0, 3, 1, 2)
        # block4
        shortcut4 = self.block4_shortcut(out)
        out = F.relu(self.block4_bn1(self.block4_conv1(out)))
        out = self.block4_bn2(self.block4_conv2(out))
        out += shortcut4
        out = F.relu(out)
        out = out.permute(0, 3, 1, 2)

        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        out = self.block10(out)
        out = out.reshape(-1, 4, 4, 4, 4, 4, 4)
        out = out.permute(0, 1, 5, 3, 4, 2, 6)
        out = out.reshape(-1, 64, 4, 4, 4)
        out = F.avg_pool3d(out, 4)
        out = out.view(-1, 64)

        # out = out.reshape(out.size(0), out.size(1)*out.size(2)*out.size(3))
        out = self.linear(out)
        return out


if __name__ == "__main__":
    import os
    from torchsummary import summary
    # twist = ResNet20_B_twist_B12(100)
    # raw_res = ResNet20_A(100)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(calc_params(twist))
    # print(calc_params(raw_res))
    model = ResNet_with_cube_block(norm="CubeNorm", num_classes=100)
    # model2 = ResNet20_A(100)
    print(summary(model, (3, 32, 32), device='cpu'))
