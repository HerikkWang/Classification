# from turtle import forward
from audioop import bias
from os import fwalk
from turtle import forward
from numpy import block, pad
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable
from torch.nn.modules.conv import Conv2d
from torchsummary import summary
from component import *
# from torch.nn.modules.linear import Linear

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet20_A(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20_A, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # block1
        self.block1_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn1 = nn.BatchNorm2d(16)
        self.block1_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn2 = nn.BatchNorm2d(16)
        self.block1_shortcut = nn.Sequential()
        # block2
        self.block2_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn1 = nn.BatchNorm2d(16)
        self.block2_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn2 = nn.BatchNorm2d(16)
        self.block2_shortcut = nn.Sequential()
        # block3
        self.block3_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn1 = nn.BatchNorm2d(16)
        self.block3_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn2 = nn.BatchNorm2d(16)
        self.block3_shortcut = nn.Sequential()
        # block4
        self.block4_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.block4_bn1 = nn.BatchNorm2d(32)
        self.block4_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block4_bn2 = nn.BatchNorm2d(32)
        self.block4_shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 32//4, 32//4), "constant", 0))
        # block5
        self.block5_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block5_bn1 = nn.BatchNorm2d(32)
        self.block5_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block5_bn2 = nn.BatchNorm2d(32)
        self.block5_shortcut = nn.Sequential()
        # block6
        self.block6_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block6_bn1 = nn.BatchNorm2d(32)
        self.block6_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block6_bn2 = nn.BatchNorm2d(32)
        self.block6_shortcut = nn.Sequential()
        # block7
        self.block7_conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.block7_bn1 = nn.BatchNorm2d(64)
        self.block7_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block7_bn2 = nn.BatchNorm2d(64)
        self.block7_shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 64//4, 64//4), "constant", 0))
        # block8
        self.block8_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block8_bn1 = nn.BatchNorm2d(64)
        self.block8_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block8_bn2 = nn.BatchNorm2d(64)
        self.block8_shortcut = nn.Sequential()
        # block9
        self.block9_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block9_bn1 = nn.BatchNorm2d(64)
        self.block9_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block9_bn2 = nn.BatchNorm2d(64)
        self.block9_shortcut = nn.Sequential()
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        # block1
        shortcut1 = self.block1_shortcut(out)
        out = F.relu(self.block1_bn1(self.block1_conv1(out)))
        # out = out.permute(0, 3, 1, 2)
        out = self.block1_bn2(self.block1_conv2(out))
        # out = out.permute(0, 3, 1, 2)
        # shortcut1 = shortcut1.permute(0, 2, 3, 1)
        out += shortcut1
        out = F.relu(out)
        # block2
        shortcut2 = self.block2_shortcut(out)
        out = F.relu(self.block2_bn1(self.block2_conv1(out)))
        # out = out.permute(0, 3, 1, 2)
        out = self.block2_bn2(self.block2_conv2(out))
        # out = out.permute(0, 3, 1, 2)
        # shortcut2 = shortcut2.permute(0, 3, 1, 2).permute(0, 3, 1, 2)
        out += shortcut2
        out = F.relu(out)
        # block3
        shortcut3 = self.block3_shortcut(out)
        out = F.relu(self.block3_bn1(self.block3_conv1(out)))
        # out = out.permute(0, 3, 1, 2)
        out = self.block3_bn2(self.block3_conv2(out))
        # out = out.permute(0, 3, 1, 2)
        # shortcut3 = shortcut3.permute(0, 2, 3, 1)
        out += shortcut3
        out = F.relu(out)
        # block4
        shortcut4 = self.block4_shortcut(out)
        out = F.relu(self.block4_bn1(self.block4_conv1(out)))
        # out = out.permute(0, 3, 1, 2)
        out = self.block4_bn2(self.block4_conv2(out))
        # out = out.permute(0, 3, 1, 2)
        # shortcut4 = shortcut4.permute(0, 3, 1, 2).permute(0, 3, 1, 2)
        out += shortcut4
        out = F.relu(out)
        # block5
        shortcut5 = self.block5_shortcut(out)
        out = F.relu(self.block5_bn1(self.block5_conv1(out)))
        # out = out.permute(0, 3, 1, 2)
        out = self.block5_bn2(self.block5_conv2(out))
        # out = out.permute(0, 3, 1, 2)
        # shortcut5 = shortcut5.permute(0, 3, 1, 2).permute(0, 3, 1, 2)
        out += shortcut5
        out = F.relu(out)
        # block6
        shortcut6 = self.block6_shortcut(out)
        out = F.relu(self.block6_bn1(self.block6_conv1(out)))
        # out = out.permute(0, 3, 1, 2)
        out = self.block6_bn2(self.block6_conv2(out))
        # out = out.permute(0, 3, 1, 2)
        # shortcut6 = shortcut6.permute(0, 3, 1, 2).permute(0, 3, 1, 2)
        out += shortcut6
        out = F.relu(out)
        # block7
        shortcut7 = self.block7_shortcut(out)
        out = F.relu(self.block7_bn1(self.block7_conv1(out)))
        # out = out.permute(0, 3, 1, 2)
        out = self.block7_bn2(self.block7_conv2(out))
        # out = out.permute(0, 3, 1, 2)
        # shortcut7 = shortcut7.permute(0, 3, 1, 2).permute(0, 3, 1, 2)
        out += shortcut7
        out = F.relu(out)
        # block8
        shortcut8 = self.block8_shortcut(out)
        out = F.relu(self.block8_bn1(self.block8_conv1(out)))
        # out = out.permute(0, 3, 1, 2)
        out = self.block8_bn2(self.block8_conv2(out))
        # out = out.permute(0, 3, 1, 2)
        # shortcut8 = shortcut8.permute(0, 3, 1, 2).permute(0, 3, 1, 2)
        out += shortcut8
        out = F.relu(out)
        # block9
        shortcut9 = self.block9_shortcut(out)
        out = F.relu(self.block9_bn1(self.block9_conv1(out)))
        # out = out.permute(0, 3, 1, 2)
        out = self.block9_bn2(self.block9_conv2(out))
        # out = out.permute(0, 3, 1, 2)
        # shortcut9 = shortcut9.permute(0, 3, 1, 2).permute(0, 3, 1, 2)
        out += shortcut9
        out = F.relu(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet20_B(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20_B, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # block1
        self.block1_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn1 = nn.BatchNorm2d(16)
        self.block1_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn2 = nn.BatchNorm2d(16)
        self.block1_shortcut = nn.Sequential()
        # block2
        self.block2_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn1 = nn.BatchNorm2d(16)
        self.block2_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn2 = nn.BatchNorm2d(16)
        self.block2_shortcut = nn.Sequential()
        # block3
        self.block3_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn1 = nn.BatchNorm2d(16)
        self.block3_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn2 = nn.BatchNorm2d(16)
        self.block3_shortcut = nn.Sequential()
        # block4
        self.block4_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.block4_bn1 = nn.BatchNorm2d(32)
        self.block4_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block4_bn2 = nn.BatchNorm2d(32)
        self.block4_shortcut = nn.Sequential(
                            nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
                            nn.BatchNorm2d(32)
                        )
        # block5
        self.block5_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block5_bn1 = nn.BatchNorm2d(32)
        self.block5_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block5_bn2 = nn.BatchNorm2d(32)
        self.block5_shortcut = nn.Sequential()
        # block6
        self.block6_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block6_bn1 = nn.BatchNorm2d(32)
        self.block6_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block6_bn2 = nn.BatchNorm2d(32)
        self.block6_shortcut = nn.Sequential()
        # block7
        self.block7_conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.block7_bn1 = nn.BatchNorm2d(64)
        self.block7_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block7_bn2 = nn.BatchNorm2d(64)
        self.block7_shortcut = nn.Sequential(
                            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
                            nn.BatchNorm2d(64)
                        )
        # block8
        self.block8_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block8_bn1 = nn.BatchNorm2d(64)
        self.block8_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block8_bn2 = nn.BatchNorm2d(64)
        self.block8_shortcut = nn.Sequential()
        # block9
        self.block9_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block9_bn1 = nn.BatchNorm2d(64)
        self.block9_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block9_bn2 = nn.BatchNorm2d(64)
        self.block9_shortcut = nn.Sequential()
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
        # block2
        shortcut2 = self.block2_shortcut(out)
        out = F.relu(self.block2_bn1(self.block2_conv1(out)))
        out = self.block2_bn2(self.block2_conv2(out))
        out += shortcut2
        out = F.relu(out)
        # block3
        shortcut3 = self.block3_shortcut(out)
        out = F.relu(self.block3_bn1(self.block3_conv1(out)))
        out = self.block3_bn2(self.block3_conv2(out))
        out += shortcut3
        out = F.relu(out)
        # block4
        shortcut4 = self.block4_shortcut(out)
        out = F.relu(self.block4_bn1(self.block4_conv1(out)))
        out = self.block4_bn2(self.block4_conv2(out))
        out += shortcut4
        out = F.relu(out)
        # block5
        shortcut5 = self.block5_shortcut(out)
        out = F.relu(self.block5_bn1(self.block5_conv1(out)))
        out = self.block5_bn2(self.block5_conv2(out))
        out += shortcut5
        out = F.relu(out)
        # block6
        shortcut6 = self.block6_shortcut(out)
        out = F.relu(self.block6_bn1(self.block6_conv1(out)))
        # out = out.permute(0, 3, 1, 2)
        out = self.block6_bn2(self.block6_conv2(out))
        # out = out.permute(0, 3, 1, 2)
        # shortcut6 = shortcut6.permute(0, 3, 1, 2).permute(0, 3, 1, 2)
        out += shortcut6
        out = F.relu(out)
        # block7
        shortcut7 = self.block7_shortcut(out)
        out = F.relu(self.block7_bn1(self.block7_conv1(out)))
        out = self.block7_bn2(self.block7_conv2(out))
        out += shortcut7
        out = F.relu(out)
        # block8
        shortcut8 = self.block8_shortcut(out)
        out = F.relu(self.block8_bn1(self.block8_conv1(out)))
        # out = out.permute(0, 3, 1, 2)
        out = self.block8_bn2(self.block8_conv2(out))
        # out = out.permute(0, 3, 1, 2)
        # shortcut8 = shortcut8.permute(0, 3, 1, 2).permute(0, 3, 1, 2)
        out += shortcut8
        out = F.relu(out)
        # block9
        shortcut9 = self.block9_shortcut(out)
        out = F.relu(self.block9_bn1(self.block9_conv1(out)))
        # out = out.permute(0, 3, 1, 2)
        out = self.block9_bn2(self.block9_conv2(out))
        # out = out.permute(0, 3, 1, 2)
        # shortcut9 = shortcut9.permute(0, 3, 1, 2).permute(0, 3, 1, 2)
        out += shortcut9
        out = F.relu(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet20_B_improved(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20_B_improved, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # block1
        self.block1_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn1 = nn.BatchNorm2d(16)
        self.block1_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn2 = nn.BatchNorm2d(16)
        self.block1_shortcut = nn.Sequential()
        # block2
        self.block2_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn1 = nn.BatchNorm2d(16)
        self.block2_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn2 = nn.BatchNorm2d(16)
        self.block2_shortcut = nn.Sequential()
        # block3
        self.block3_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn1 = nn.BatchNorm2d(16)
        self.block3_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn2 = nn.BatchNorm2d(16)
        self.block3_shortcut = nn.Sequential()
        # block4
        self.block4_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.block4_bn1 = nn.BatchNorm2d(32)
        self.block4_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block4_bn2 = nn.BatchNorm2d(32)
        self.block4_shortcut = nn.Sequential(
                            nn.AvgPool2d((2, 2)),
                            nn.Conv2d(16, 32, kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(32)
                        )
        # block5
        self.block5_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block5_bn1 = nn.BatchNorm2d(32)
        self.block5_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block5_bn2 = nn.BatchNorm2d(32)
        self.block5_shortcut = nn.Sequential()
        # block6
        self.block6_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block6_bn1 = nn.BatchNorm2d(32)
        self.block6_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block6_bn2 = nn.BatchNorm2d(32)
        self.block6_shortcut = nn.Sequential()
        # block7
        self.block7_conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.block7_bn1 = nn.BatchNorm2d(64)
        self.block7_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block7_bn2 = nn.BatchNorm2d(64)
        self.block7_shortcut = nn.Sequential(
                            nn.AvgPool2d((2, 2)),
                            nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(64)
                        )
        # block8
        self.block8_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block8_bn1 = nn.BatchNorm2d(64)
        self.block8_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block8_bn2 = nn.BatchNorm2d(64)
        self.block8_shortcut = nn.Sequential()
        # block9
        self.block9_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block9_bn1 = nn.BatchNorm2d(64)
        self.block9_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block9_bn2 = nn.BatchNorm2d(64)
        self.block9_shortcut = nn.Sequential()
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
        # block2
        shortcut2 = self.block2_shortcut(out)
        out = F.relu(self.block2_bn1(self.block2_conv1(out)))
        out = self.block2_bn2(self.block2_conv2(out))
        out += shortcut2
        out = F.relu(out)
        # block3
        shortcut3 = self.block3_shortcut(out)
        out = F.relu(self.block3_bn1(self.block3_conv1(out)))
        out = self.block3_bn2(self.block3_conv2(out))
        out += shortcut3
        out = F.relu(out)
        # block4
        shortcut4 = self.block4_shortcut(out)
        out = F.relu(self.block4_bn1(self.block4_conv1(out)))
        out = self.block4_bn2(self.block4_conv2(out))
        out += shortcut4
        out = F.relu(out)
        # block5
        shortcut5 = self.block5_shortcut(out)
        out = F.relu(self.block5_bn1(self.block5_conv1(out)))
        out = self.block5_bn2(self.block5_conv2(out))
        out += shortcut5
        out = F.relu(out)
        # block6
        shortcut6 = self.block6_shortcut(out)
        out = F.relu(self.block6_bn1(self.block6_conv1(out)))
        # out = out.permute(0, 3, 1, 2)
        out = self.block6_bn2(self.block6_conv2(out))
        # out = out.permute(0, 3, 1, 2)
        # shortcut6 = shortcut6.permute(0, 3, 1, 2).permute(0, 3, 1, 2)
        out += shortcut6
        out = F.relu(out)
        # block7
        shortcut7 = self.block7_shortcut(out)
        out = F.relu(self.block7_bn1(self.block7_conv1(out)))
        out = self.block7_bn2(self.block7_conv2(out))
        out += shortcut7
        out = F.relu(out)
        # block8
        shortcut8 = self.block8_shortcut(out)
        out = F.relu(self.block8_bn1(self.block8_conv1(out)))
        # out = out.permute(0, 3, 1, 2)
        out = self.block8_bn2(self.block8_conv2(out))
        # out = out.permute(0, 3, 1, 2)
        # shortcut8 = shortcut8.permute(0, 3, 1, 2).permute(0, 3, 1, 2)
        out += shortcut8
        out = F.relu(out)
        # block9
        shortcut9 = self.block9_shortcut(out)
        out = F.relu(self.block9_bn1(self.block9_conv1(out)))
        # out = out.permute(0, 3, 1, 2)
        out = self.block9_bn2(self.block9_conv2(out))
        # out = out.permute(0, 3, 1, 2)
        # shortcut9 = shortcut9.permute(0, 3, 1, 2).permute(0, 3, 1, 2)
        out += shortcut9
        out = F.relu(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet20_B_bkplug(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20_B_bkplug, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # block1
        self.block1_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn1 = nn.BatchNorm2d(16)
        self.block1_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn2 = nn.BatchNorm2d(16)
        self.block1_shortcut = nn.Sequential()
        # block2
        self.block2_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn1 = nn.BatchNorm2d(16)
        self.block2_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn2 = nn.BatchNorm2d(16)
        self.block2_shortcut = nn.Sequential()
        # block3
        self.block3_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn1 = nn.BatchNorm2d(16)
        self.block3_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn2 = nn.BatchNorm2d(16)
        self.block3_shortcut = nn.Sequential()
        # block4
        self.block4_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.block4_bn1 = nn.BatchNorm2d(32)
        self.block4_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block4_bn2 = nn.BatchNorm2d(32)
        self.block4_shortcut = nn.Sequential(
                            nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
                            nn.BatchNorm2d(32)
                        )
        # block5
        self.block5_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block5_bn1 = nn.BatchNorm2d(32)
        self.block5_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block5_bn2 = nn.BatchNorm2d(32)
        self.block5_shortcut = nn.Sequential()
        # block6
        self.block6_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block6_bn1 = nn.BatchNorm2d(32)
        self.block6_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block6_bn2 = nn.BatchNorm2d(32)
        self.block6_shortcut = nn.Sequential()
        # block7
        self.block7_conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.block7_bn1 = nn.BatchNorm2d(64)
        self.block7_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block7_bn2 = nn.BatchNorm2d(64)
        self.block7_shortcut = nn.Sequential(
                            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
                            nn.BatchNorm2d(64)
                        )
        # block8
        self.block8_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block8_bn1 = nn.BatchNorm2d(64)
        self.block8_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block8_bn2 = nn.BatchNorm2d(64)
        self.block8_shortcut = nn.Sequential()
        # block9
        self.block9_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block9_bn1 = nn.BatchNorm2d(64)
        self.block9_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block9_bn2 = nn.BatchNorm2d(64)
        self.block9_shortcut = nn.Sequential()
        self.linear = nn.Linear(64, num_classes)
        self.plug1 = shuffle_plug(32, 32, 32, 32, 16)
        # self.plug2 = nn.Sequential()
        self.plug3 = nn.Sequential()
        # self.plug4 = nn.Sequential()
        self.plug5 = nn.Sequential()
        self.plug6 = nn.Sequential()
        self.plug7 = nn.Sequential()
        self.plug8 = nn.Sequential()
        # self.plug9 = nn.Sequential()
        self.plug2 = shuffle_plug(32, 32, 32, 32, 16)
        # self.plug3 = shuffle_plug(32, 32, 32, 32, 16)
        self.plug4 = shuffle_plug(16, 16, 16, 16, 32)
        # self.plug5 = shuffle_plug(16, 16, 16, 16, 32)
        # self.plug6 = shuffle_plug(16, 16, 16, 16, 32)
        # self.plug7 = shuffle_plug(8, 8, 8, 8, 64)
        # self.plug8 = shuffle_plug(8, 8, 8, 8, 64)
        self.plug0 = shuffle_plug(32, 32, 32, 32, 16)
        # self.plug9 = shuffle_plug(8, 8, 8, 8, 64)
        self.apply(_weights_init)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.plug0(out)
        # block1
        shortcut1 = self.block1_shortcut(out)
        out = F.relu(self.block1_bn1(self.block1_conv1(out)))
        out = self.block1_bn2(self.block1_conv2(out))
        out += shortcut1
        out = F.relu(out)
        out = self.plug1(out)
        # block2
        shortcut2 = self.block2_shortcut(out)
        out = F.relu(self.block2_bn1(self.block2_conv1(out)))
        out = self.block2_bn2(self.block2_conv2(out))
        out += shortcut2
        out = F.relu(out)
        out = self.plug2(out)
        # block3
        shortcut3 = self.block3_shortcut(out)
        out = F.relu(self.block3_bn1(self.block3_conv1(out)))
        out = self.block3_bn2(self.block3_conv2(out))
        out += shortcut3
        out = F.relu(out)
        out = self.plug3(out)
        # block4
        shortcut4 = self.block4_shortcut(out)
        out = F.relu(self.block4_bn1(self.block4_conv1(out)))
        out = self.block4_bn2(self.block4_conv2(out))
        out += shortcut4
        out = F.relu(out)
        out = self.plug4(out)
        # block5
        shortcut5 = self.block5_shortcut(out)
        out = F.relu(self.block5_bn1(self.block5_conv1(out)))
        out = self.block5_bn2(self.block5_conv2(out))
        out += shortcut5
        out = F.relu(out)
        out = self.plug5(out)
        # block6
        shortcut6 = self.block6_shortcut(out)
        out = F.relu(self.block6_bn1(self.block6_conv1(out)))
        # out = out.permute(0, 3, 1, 2)
        out = self.block6_bn2(self.block6_conv2(out))
        out += shortcut6
        out = F.relu(out)
        out = self.plug6(out)
        # block7
        shortcut7 = self.block7_shortcut(out)
        out = F.relu(self.block7_bn1(self.block7_conv1(out)))
        out = self.block7_bn2(self.block7_conv2(out))
        out += shortcut7
        out = F.relu(out)
        out = self.plug7(out)
        # block8
        shortcut8 = self.block8_shortcut(out)
        out = F.relu(self.block8_bn1(self.block8_conv1(out)))
        out = self.block8_bn2(self.block8_conv2(out))
        out += shortcut8
        out = F.relu(out)
        out = self.plug8(out)
        # block9
        shortcut9 = self.block9_shortcut(out)
        out = F.relu(self.block9_bn1(self.block9_conv1(out)))
        out = self.block9_bn2(self.block9_conv2(out))
        out += shortcut9
        out = F.relu(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet20_B_bkplug_2(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20_B_bkplug_2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # block1
        self.block1_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn1 = nn.BatchNorm2d(16)
        self.block1_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn2 = nn.BatchNorm2d(16)
        self.block1_shortcut = nn.Sequential()
        # block2
        self.block2_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn1 = nn.BatchNorm2d(16)
        self.block2_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn2 = nn.BatchNorm2d(16)
        self.block2_shortcut = nn.Sequential()
        # block3
        self.block3_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn1 = nn.BatchNorm2d(16)
        self.block3_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn2 = nn.BatchNorm2d(16)
        self.block3_shortcut = nn.Sequential()
        # block4
        self.block4_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.block4_bn1 = nn.BatchNorm2d(32)
        self.block4_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block4_bn2 = nn.BatchNorm2d(32)
        self.block4_shortcut = nn.Sequential(
                            nn.AvgPool2d((2, 2)),
                            nn.Conv2d(16, 32, kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(32)
                        )
        # block5
        self.block5_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block5_bn1 = nn.BatchNorm2d(32)
        self.block5_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block5_bn2 = nn.BatchNorm2d(32)
        self.block5_shortcut = nn.Sequential()
        # block6
        self.block6_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block6_bn1 = nn.BatchNorm2d(32)
        self.block6_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block6_bn2 = nn.BatchNorm2d(32)
        self.block6_shortcut = nn.Sequential()
        # block7
        self.block7_conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.block7_bn1 = nn.BatchNorm2d(64)
        self.block7_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block7_bn2 = nn.BatchNorm2d(64)
        self.block7_shortcut = nn.Sequential(
                            nn.AvgPool2d((2, 2)),
                            nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(64)
                        )
        # block8
        self.block8_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block8_bn1 = nn.BatchNorm2d(64)
        self.block8_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block8_bn2 = nn.BatchNorm2d(64)
        self.block8_shortcut = nn.Sequential()
        # block9
        self.block9_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block9_bn1 = nn.BatchNorm2d(64)
        self.block9_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block9_bn2 = nn.BatchNorm2d(64)
        self.block9_shortcut = nn.Sequential()
        self.linear = nn.Linear(64, num_classes)
        self.plug1 = shuffle_plug_3(32, 32, 16)
        self.plug2 = nn.Sequential()
        self.plug3 = nn.Sequential()
        self.plug4 = nn.Sequential()
        self.plug5 = nn.Sequential()
        self.plug6 = nn.Sequential()
        self.plug7 = nn.Sequential()
        self.plug8 = nn.Sequential()
        self.plug9 = nn.Sequential()
        self.plug2 = shuffle_plug_3(32, 32, 16)
        self.plug0 = nn.Sequential()
        self.plug3 = shuffle_plug_3(32, 32, 16)
        # self.plug4 = shuffle_plug_3(16, 16, 32)
        # self.plug5 = shuffle_plug_3(16, 16, 32)
        # self.plug6 = shuffle_plug_3(16, 16, 32)
        # self.plug7 = shuffle_plug_3(8, 8, 64)
        # self.plug8 = shuffle_plug_3(8, 8, 64)
        # self.plug0 = shuffle_plug_3(32, 32, 16)
        # self.plug9 = shuffle_plug(8, 8, 8, 8, 64)
        self.apply(_weights_init)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        # out = self.plug0(out)
        # out = self.plug1(out)
        # block1
        shortcut1 = self.block1_shortcut(out)
        out = F.relu(self.block1_bn1(self.block1_conv1(out)))
        out = self.block1_bn2(self.block1_conv2(out))
        out += shortcut1
        out = F.relu(out)
        out = self.plug1(out)
        # block2
        shortcut2 = self.block2_shortcut(out)
        out = F.relu(self.block2_bn1(self.block2_conv1(out)))
        out = self.block2_bn2(self.block2_conv2(out))
        out += shortcut2
        out = F.relu(out)
        out = self.plug2(out)
        # block3
        shortcut3 = self.block3_shortcut(out)
        out = F.relu(self.block3_bn1(self.block3_conv1(out)))
        out = self.block3_bn2(self.block3_conv2(out))
        out += shortcut3
        out = F.relu(out)
        out = self.plug3(out)
        # block4
        shortcut4 = self.block4_shortcut(out)
        out = F.relu(self.block4_bn1(self.block4_conv1(out)))
        out = self.block4_bn2(self.block4_conv2(out))
        out += shortcut4
        out = F.relu(out)
        out = self.plug4(out)
        # block5
        shortcut5 = self.block5_shortcut(out)
        out = F.relu(self.block5_bn1(self.block5_conv1(out)))
        out = self.block5_bn2(self.block5_conv2(out))
        out += shortcut5
        out = F.relu(out)
        out = self.plug5(out)
        # block6
        shortcut6 = self.block6_shortcut(out)
        out = F.relu(self.block6_bn1(self.block6_conv1(out)))
        # out = out.permute(0, 3, 1, 2)
        out = self.block6_bn2(self.block6_conv2(out))
        out += shortcut6
        out = F.relu(out)
        out = self.plug6(out)
        # block7
        shortcut7 = self.block7_shortcut(out)
        out = F.relu(self.block7_bn1(self.block7_conv1(out)))
        out = self.block7_bn2(self.block7_conv2(out))
        out += shortcut7
        out = F.relu(out)
        out = self.plug7(out)
        # block8
        shortcut8 = self.block8_shortcut(out)
        out = F.relu(self.block8_bn1(self.block8_conv1(out)))
        out = self.block8_bn2(self.block8_conv2(out))
        out += shortcut8
        out = F.relu(out)
        out = self.plug8(out)
        # block9
        shortcut9 = self.block9_shortcut(out)
        out = F.relu(self.block9_bn1(self.block9_conv1(out)))
        out = self.block9_bn2(self.block9_conv2(out))
        out += shortcut9
        out = F.relu(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet20_B_plug(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20_B_plug, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # block1
        self.block1_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn1 = nn.BatchNorm2d(16)
        self.plug1 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.plug1_norm = nn.BatchNorm2d(16)
        self.block1_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn2 = nn.BatchNorm2d(16)
        self.block1_shortcut = nn.Sequential()
        # block2
        self.plug2 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.plug2_norm = nn.BatchNorm2d(16)
        self.block2_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn1 = nn.BatchNorm2d(16)
        self.plug3 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.plug3_norm = nn.BatchNorm2d(16)
        self.block2_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn2 = nn.BatchNorm2d(16)
        self.block2_shortcut = nn.Sequential()
        # block3
        self.block3_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn1 = nn.BatchNorm2d(16)
        self.block3_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn2 = nn.BatchNorm2d(16)
        self.block3_shortcut = nn.Sequential()
        # block4
        self.block4_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.block4_bn1 = nn.BatchNorm2d(32)
        self.block4_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block4_bn2 = nn.BatchNorm2d(32)
        self.block4_shortcut = nn.Sequential(
                            nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
                            nn.BatchNorm2d(32)
                        )
        # block5
        self.block5_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block5_bn1 = nn.BatchNorm2d(32)
        self.block5_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block5_bn2 = nn.BatchNorm2d(32)
        self.block5_shortcut = nn.Sequential()
        # block6
        self.block6_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block6_bn1 = nn.BatchNorm2d(32)
        self.block6_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block6_bn2 = nn.BatchNorm2d(32)
        self.block6_shortcut = nn.Sequential()
        # block7
        self.block7_conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.block7_bn1 = nn.BatchNorm2d(64)
        self.block7_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block7_bn2 = nn.BatchNorm2d(64)
        self.block7_shortcut = nn.Sequential(
                            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
                            nn.BatchNorm2d(64)
                        )
        # block8
        self.block8_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block8_bn1 = nn.BatchNorm2d(64)
        self.block8_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block8_bn2 = nn.BatchNorm2d(64)
        self.block8_shortcut = nn.Sequential()
        # block9
        self.block9_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block9_bn1 = nn.BatchNorm2d(64)
        self.block9_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block9_bn2 = nn.BatchNorm2d(64)
        self.block9_shortcut = nn.Sequential()
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        # block1
        shortcut1 = self.block1_shortcut(out)
        out = F.relu(self.block1_bn1(self.block1_conv1(out)))
        out = out.permute(position_coding("NCWH", "NHCW"))
        out = self.plug1(out)
        out = out.permute(position_coding("NHCW", "NCWH"))
        # out = F.relu(self.plug1_norm(out))
        out = self.block1_bn2(self.block1_conv2(out))
        out += shortcut1
        out = F.relu(out)
        # block2
        shortcut2 = self.block2_shortcut(out)
        out = out.permute(position_coding("NCWH", "NHCW"))
        out = self.plug2(out)
        out = out.permute(position_coding("NHCW", "NCWH"))
        # out = F.relu(self.plug2_norm(out))
        out = F.relu(self.block2_bn1(self.block2_conv1(out)))
        out = out.permute(position_coding("NCWH", "NWHC"))
        out = self.plug3(out)
        out = out.permute(position_coding("NWHC", "NCWH"))
        # out = F.relu(self.plug3_norm(out))
        out = self.block2_bn2(self.block2_conv2(out))
        out += shortcut2
        out = F.relu(out)
        # block3
        shortcut3 = self.block3_shortcut(out)
        out = F.relu(self.block3_bn1(self.block3_conv1(out)))
        out = self.block3_bn2(self.block3_conv2(out))
        out += shortcut3
        out = F.relu(out)
        # block4
        shortcut4 = self.block4_shortcut(out)
        out = F.relu(self.block4_bn1(self.block4_conv1(out)))
        out = self.block4_bn2(self.block4_conv2(out))
        out += shortcut4
        out = F.relu(out)
        # block5
        shortcut5 = self.block5_shortcut(out)
        out = F.relu(self.block5_bn1(self.block5_conv1(out)))
        out = self.block5_bn2(self.block5_conv2(out))
        out += shortcut5
        out = F.relu(out)
        # block6
        shortcut6 = self.block6_shortcut(out)
        out = F.relu(self.block6_bn1(self.block6_conv1(out)))
        # out = out.permute(0, 3, 1, 2)
        out = self.block6_bn2(self.block6_conv2(out))
        # out = out.permute(0, 3, 1, 2)
        # shortcut6 = shortcut6.permute(0, 3, 1, 2).permute(0, 3, 1, 2)
        out += shortcut6
        out = F.relu(out)
        # block7
        shortcut7 = self.block7_shortcut(out)
        out = F.relu(self.block7_bn1(self.block7_conv1(out)))
        out = self.block7_bn2(self.block7_conv2(out))
        out += shortcut7
        out = F.relu(out)
        # block8
        shortcut8 = self.block8_shortcut(out)
        out = F.relu(self.block8_bn1(self.block8_conv1(out)))
        # out = out.permute(0, 3, 1, 2)
        out = self.block8_bn2(self.block8_conv2(out))
        # out = out.permute(0, 3, 1, 2)
        # shortcut8 = shortcut8.permute(0, 3, 1, 2).permute(0, 3, 1, 2)
        out += shortcut8
        out = F.relu(out)
        # block9
        shortcut9 = self.block9_shortcut(out)
        out = F.relu(self.block9_bn1(self.block9_conv1(out)))
        # out = out.permute(0, 3, 1, 2)
        out = self.block9_bn2(self.block9_conv2(out))
        # out = out.permute(0, 3, 1, 2)
        # shortcut9 = shortcut9.permute(0, 3, 1, 2).permute(0, 3, 1, 2)
        out += shortcut9
        out = F.relu(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class shuffle_plug(nn.Module):
    def __init__(self, in_ch1, out_ch1, in_ch2, out_ch2, bn_ch, norm1=True, norm2=True, relu1=False, relu2=True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch1, out_ch1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_ch2, out_ch2, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu1 = relu1
        self.relu2 = relu2
        self.linear = nn.Conv2d(bn_ch, bn_ch, kernel_size=1, stride=1, padding=0, bias=False)
        if norm1:
            self.bn1 = nn.BatchNorm2d(bn_ch)
        else:
            self.bn1 = nn.Sequential()
        
        if norm2:
            self.bn2 = nn.BatchNorm2d(bn_ch)
        else:
            self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
    
    def forward(self, x):
        shortcut = self.shortcut(x)
        out = x.permute(position_coding("NCWH", "NHCW"))
        out = self.conv1(out)
        if self.relu1:
            out = F.relu(self.bn1(out.permute(position_coding("NHCW", "NCWH"))))
        else:
            out = self.bn1(out.permute(position_coding("NHCW", "NCWH")))
        out = out.permute(position_coding("NCWH", "NWHC"))
        out = self.conv2(out)
        out = out.permute(position_coding("NWHC", "NCWH"))
        out = self.bn2(out)
        if self.relu2:
            out = F.relu(out + shortcut)
        else:
            out = self.linear(out + shortcut)
        # out = F.relu()
        return out

class shuffle_plug_2(nn.Module):
    def __init__(self, in_ch1, out_ch1, bn_ch, pos_code1, pos_code2, norm1=True, relu1=True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch1, out_ch1, kernel_size=1, stride=1, padding=0)
        # self.conv2 = nn.Conv2d(in_ch2, out_ch2, kernel_size=1, stride=1, padding=0)
        self.relu1 = relu1
        # self.relu2 = relu2
        # self.linear = nn.Conv2d(bn_ch, bn_ch, kernel_size=1, stride=1, padding=0)
        if norm1:
            self.bn1 = nn.BatchNorm2d(bn_ch)
        else:
            self.bn1 = nn.Sequential()
        self.pos_code1 = pos_code1
        self.pos_code2 = pos_code2
        
        # if norm2:
        #     self.bn2 = nn.BatchNorm2d(bn_ch)
        # else:
        #     self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
    
    def forward(self, x):
        # shortcut = self.shortcut(x)
        out = x.permute(self.pos_code1)
        out = self.conv1(out)
        if self.relu1:
            out = F.relu(self.shortcut(x) + self.bn1(out.permute(self.pos_code2)))
        else:
            out = self.bn1(out.permute(self.pos_code2))
        # out = out.permute(position_coding("NCWH", "NWHC"))
        # out = self.conv2(out)
        # out = out.permute(position_coding("NWHC", "NCWH"))
        # out = self.bn2(out)
        # if self.relu2:
        #     out = F.relu(out + shortcut)
        # else:
        #     out = self.linear(out + shortcut)
        # out = F.relu()
        return out

def calc_params(model):
    return sum(p.numel() for p in model.parameters())

def position_coding(pos1, pos2):
    char_dict = {}
    for i in range(len(pos1)):
        char_dict[pos1[i]] = i
    
    code = []
    for i in range(len(pos2)):
        code.append(char_dict[pos2[i]])
    
    return tuple(code)

if __name__ == "__main__":
    import os
    from torchsummary import summary
    # twist = ResNet20_B_twist_B12(100)
    # raw_res = ResNet20_A(100)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(calc_params(twist))
    # print(calc_params(raw_res))
    model = ResNet20_B_bkplug_2(num_classes=100)
    model2 = ResNet20_B(100)
    print(summary(model, (3, 32, 32), device='cpu'))
    # print(summary(model2, (3, 32, 32), device='cpu'))
    # print(summary(model2, (3, 32, 32), device='cpu'))
    # print(summary(Cube_net(100), (3, 32, 32), device='cpu'))
    # print(calc_params(model))
    # print(calc_params(ResNet20_B(100)))
    # print(calc_params(Cube_net_conv1(100)))
    # print(position_coding("NCWH", "NHCW"))




