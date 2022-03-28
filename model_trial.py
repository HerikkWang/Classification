from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
from utils.functions import position_coding, _weights_init
from component import *
from component import ConvMixer

class model_trial_v0(nn.Module):
    def __init__(self, num_classes=10):
        super(model_trial_v0, self).__init__()
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
        # self.plug2 = nn.Sequential()
        # self.plug3 = nn.Sequential()
        self.plug4 = nn.Sequential()
        self.plug5 = nn.Sequential()
        self.plug6 = nn.Sequential()
        self.plug7 = nn.Sequential()
        self.plug8 = nn.Sequential()
        self.plug9 = nn.Sequential()
        self.plug2 = shuffle_plug_3(32, 32, 16)
        self.plug0 = nn.Sequential()
        self.plug3 = shuffle_plug_3(32, 32, 16)
        # self.plug4 = shuffle_plug_3(32, 32, 16)
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
        # shortcut4 = self.block4_shortcut(out)
        shortcut4 = self.block4_shortcut(F.avg_pool2d(out, 2))
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
        shortcut7 = self.block7_shortcut(F.avg_pool2d(out, 2))
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

class ResNet56_modified(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.block1 = BasicBlock(16, 16)
        self.block2 = BasicBlock(16, 16)
        self.block3 = BasicBlock(16, 16)
        self.block4 = BasicBlock(16, 16)
        self.block5 = BasicBlock(16, 16)
        self.block6 = BasicBlock(16, 16)
        self.block7 = BasicBlock(16, 16)
        self.block8 = BasicBlock(16, 16)
        self.block9 = BasicBlock(16, 16)
        self.block10 = BasicBlock(16, 32, 3, 2)
        self.block11 = BasicBlock(32, 32)
        self.block12 = BasicBlock(32, 32)
        self.block13 = BasicBlock(32, 32)
        self.block14 = BasicBlock(32, 32)
        self.block15 = BasicBlock(32, 32)
        self.block16 = BasicBlock(32, 32)
        self.block17 = BasicBlock(32, 32)
        self.block18 = BasicBlock(32, 32)
        self.block19 = BasicBlock(32, 64, 3, 2)
        self.block20 = BasicBlock(64, 64)
        self.block21 = BasicBlock(64, 64)
        self.block22 = BasicBlock(64, 64)
        self.block23 = BasicBlock(64, 64)
        self.block24 = BasicBlock(64, 64)
        self.block25 = BasicBlock(64, 64)
        self.block26 = BasicBlock(64, 64)
        self.block27 = BasicBlock(64, 64)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)
        # self.plug1 = shuffle_plug_with_cycle(32, 16)
        self.plug1 = adjustable_plug(1, 32, 16)
        self.plug2 = adjustable_plug(2, 32, 16)
        # self.plug2 = shuffle_plug_with_cycle(32, 16)
        self.plug3 = adjustable_plug(3, 32, 16)
        # self.plug2 = shuffle_plug_padding3(32, 32, 16, kernel_size=7)
        # self.plug3 = shuffle_plug_padding3(32, 32, 16, kernel_size=7)
        # self.plug1 = CycleTwist2(16, 32, (1, 5))
        # self.plug2 = CycleTwist(16, 32, (1, 5))
        # self.plug3 = shuffle_plug_3(32, 32, 16)
        self.plug4 = adjustable_plug(4, 32, 16)
        # self.plug5 = shuffle_plug_padding3(32, 32, 16, kernel_size=7)
        # self.plug6 = shuffle_plug_padding3(32, 32, 16, kernel_size=7)
        # self.plug7 = shuffle_plug_padding3(32, 32, 16, kernel_size=7)
        # self.plug8 = shuffle_plug_padding3(32, 32, 16, kernel_size=7)
        # self.plug9 = shuffle_plug_padding3(32, 32, 16, kernel_size=7)
        # self.plug10 = shuffle_plug_3(16, 16, 32)

    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.block1(out)
        out = self.plug1(out)
        
        
        # out = self.plug2(out)
        out = self.block2(out)
        # out = self.plug2(out)
        
        out = self.block3(out)
        # out = self.plug3(out)
        out = self.block4(out)
        # out = self.plug4(out)
        out = self.block5(out)
        # out = self.plug5(out)
        out = self.block6(out)
        # out = self.plug6(out)
        out = self.block7(out)
        # out = self.plug7(out)
        out = self.block8(out)
        # out = self.plug8(out)
        out = self.block9(out)
        # out = self.plug9(out)
        # out = self.plug9(out)
        out = self.block10(out)
        # out = self.plug10(out)
        out = self.block11(out)
        out = self.block12(out)
        out = self.block13(out)
        out = self.block14(out)
        out = self.block15(out)
        out = self.block16(out)
        out = self.block17(out)
        out = self.block18(out)
        out = self.block19(out)
        out = self.block20(out)
        out = self.block21(out)
        out = self.block22(out)
        out = self.block23(out)
        out = self.block24(out)
        out = self.block25(out)
        out = self.block26(out)
        out = self.block27(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet42_modified(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.block1 = BasicBlock(16, 16)
        self.block2 = BasicBlock(16, 16)
        self.block3 = BasicBlock(16, 16)
        self.block4 = BasicBlock(16, 16)
        self.block5 = BasicBlock(16, 16)
        self.block6 = BasicBlock(16, 16)
        self.block7 = BasicBlock(16, 16)
        self.block8 = BasicBlock(16, 16)
        self.block9 = BasicBlock(16, 16)
        self.block10 = BasicBlock(16, 32, 3, 2)
        self.block11 = BasicBlock(32, 32)
        self.block12 = BasicBlock(32, 32)
        self.block13 = BasicBlock(32, 32)
        self.block14 = BasicBlock(32, 32)
        self.block15 = BasicBlock(32, 32)
        self.block16 = BasicBlock(32, 32)
        self.block17 = BasicBlock(32, 32)
        self.block18 = BasicBlock(32, 32)
        self.block19 = BasicBlock(32, 64, 2)
        self.block20 = BasicBlock(64, 64)
        # self.block21 = BasicBlock(64, 64)
        # self.block22 = BasicBlock(64, 64)
        # self.block23 = BasicBlock(64, 64)
        # self.block24 = BasicBlock(64, 64)
        # self.block25 = BasicBlock(64, 64)
        # self.block26 = BasicBlock(64, 64)
        # self.block27 = BasicBlock(64, 64)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)
        self.plug1 = shuffle_plug_3(32, 32, 16)
        self.plug2 = shuffle_plug_3(32, 32, 16)
        self.plug3 = shuffle_plug_3(32, 32, 16)
        # self.plug4 = shuffle_plug_3(32, 32, 16)
        # self.plug5 = shuffle_plug_3(32, 32, 16)
        # self.plug6 = shuffle_plug_3(32, 32, 16)
        # self.plug7 = shuffle_plug_3(32, 32, 16)
        # self.plug8 = shuffle_plug_3(32, 32, 16)
        # self.plug9 = shuffle_plug_3(32, 32, 16)

    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.block1(out)
        out = self.plug1(out)
        out = self.block2(out)
        out = self.plug2(out)
        out = self.block3(out)
        out = self.plug3(out)
        out = self.block4(out)
        # out = self.plug4(out)
        out = self.block5(out)
        # out = self.plug5(out)
        out = self.block6(out)
        # out = self.plug6(out)
        out = self.block7(out)
        # out = self.plug7(out)
        out = self.block8(out)
        # out = self.plug8(out)
        out = self.block9(out)
        # out = self.plug9(out)
        out = self.block10(out)
        out = self.block11(out)
        out = self.block12(out)
        out = self.block13(out)
        out = self.block14(out)
        out = self.block15(out)
        out = self.block16(out)
        out = self.block17(out)
        out = self.block18(out)
        out = self.block19(out)
        out = self.block20(out)
        # out = self.block21(out)
        # out = self.block22(out)
        # out = self.block23(out)
        # out = self.block24(out)
        # out = self.block25(out)
        # out = self.block26(out)
        # out = self.block27(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class test_model_1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.block1 = TestBlock(16, 16, 32, 1, True)
        self.block2 = TestBlock(16, 16, 32)
        self.block3 = TestBlock(16, 16, 32)
        self.block4 = TestBlock(16, 32, 32, stride=2)
        self.block5 = TestBlock(32, 32, 16)
        self.block6 = TestBlock(32, 32, 16)
        self.block7 = TestBlock(32, 64, 16, stride=2)
        self.block8 = TestBlock(64, 64, 8)
        self.block9 = TestBlock(64, 64, 8)

        # self.block21 = BasicBlock(64, 64)
        # self.block22 = BasicBlock(64, 64)
        # self.block23 = BasicBlock(64, 64)
        # self.block24 = BasicBlock(64, 64)
        # self.block25 = BasicBlock(64, 64)
        # self.block26 = BasicBlock(64, 64)
        # self.block27 = BasicBlock(64, 64)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class model_trial_v1(nn.Module):
    def __init__(self, num_classes=10):
        super(model_trial_v1, self).__init__()
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
        self.plug1 = adjustable_plug(1, 32, 16)
        self.plug2 = adjustable_plug(2, 32, 16)
        # self.plug3 = shuffle_plug_4(32, 32, 16)
        # self.plug4 = CycleTwist(32, 16, (1, 5))
        # self.plug5 = CycleTwist(32, 16, (1, 5))
        self.plug6 = nn.Sequential()
        self.plug7 = nn.Sequential()
        self.plug8 = nn.Sequential()
        self.plug9 = nn.Sequential()
        # self.plug2 = nn.Sequential()
        self.plug0 = adjustable_plug(0, 32, 16)
        self.plug3 = adjustable_plug(3, 32, 16)
        self.plug4 = nn.Sequential()
        self.plug5 = nn.Sequential()
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
        # shortcut4 = self.block4_shortcut(out)
        shortcut4 = self.block4_shortcut(F.avg_pool2d(out, 2))
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
        shortcut7 = self.block7_shortcut(F.avg_pool2d(out, 2))
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

class softmax_conv_model(nn.Module):
    def __init__(self, num_classes=10):
        super(softmax_conv_model, self).__init__()
        self.conv1 = SoftMaxConv(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # block1
        self.block1_conv1 = SoftMaxConv(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn1 = nn.BatchNorm2d(16)
        self.block1_conv2 = SoftMaxConv(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn2 = nn.BatchNorm2d(16)
        self.block1_shortcut = nn.Sequential()
        # block2
        self.block2_conv1 = SoftMaxConv(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn1 = nn.BatchNorm2d(16)
        self.block2_conv2 = SoftMaxConv(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn2 = nn.BatchNorm2d(16)
        self.block2_shortcut = nn.Sequential()
        # block3
        self.block3_conv1 = SoftMaxConv(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn1 = nn.BatchNorm2d(16)
        self.block3_conv2 = SoftMaxConv(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn2 = nn.BatchNorm2d(16)
        self.block3_shortcut = nn.Sequential()
        # block4
        self.block4_conv1 = SoftMaxConv(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.block4_bn1 = nn.BatchNorm2d(32)
        self.block4_conv2 = SoftMaxConv(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block4_bn2 = nn.BatchNorm2d(32)
        self.block4_shortcut = nn.Sequential(
                            SoftMaxConv(16, 32, kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(32)
                        )
        # block5
        self.block5_conv1 = SoftMaxConv(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block5_bn1 = nn.BatchNorm2d(32)
        self.block5_conv2 = SoftMaxConv(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block5_bn2 = nn.BatchNorm2d(32)
        self.block5_shortcut = nn.Sequential()
        # block6
        self.block6_conv1 = SoftMaxConv(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block6_bn1 = nn.BatchNorm2d(32)
        self.block6_conv2 = SoftMaxConv(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block6_bn2 = nn.BatchNorm2d(32)
        self.block6_shortcut = nn.Sequential()
        # block7
        self.block7_conv1 = SoftMaxConv(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.block7_bn1 = nn.BatchNorm2d(64)
        self.block7_conv2 = SoftMaxConv(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block7_bn2 = nn.BatchNorm2d(64)
        self.block7_shortcut = nn.Sequential(
                            SoftMaxConv(32, 64, kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(64)
                        )
        # block8
        self.block8_conv1 = SoftMaxConv(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block8_bn1 = nn.BatchNorm2d(64)
        self.block8_conv2 = SoftMaxConv(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block8_bn2 = nn.BatchNorm2d(64)
        self.block8_shortcut = nn.Sequential()
        # block9
        self.block9_conv1 = SoftMaxConv(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block9_bn1 = nn.BatchNorm2d(64)
        self.block9_conv2 = SoftMaxConv(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
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
        # shortcut4 = self.block4_shortcut(out)
        shortcut4 = self.block4_shortcut(F.avg_pool2d(out, 2))
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
        out += shortcut6
        out = F.relu(out)
        # block7
        shortcut7 = self.block7_shortcut(F.avg_pool2d(out, 2))
        out = F.relu(self.block7_bn1(self.block7_conv1(out)))
        out = self.block7_bn2(self.block7_conv2(out))
        out += shortcut7
        out = F.relu(out)
        # block8
        shortcut8 = self.block8_shortcut(out)
        out = F.relu(self.block8_bn1(self.block8_conv1(out)))
        out = self.block8_bn2(self.block8_conv2(out))
        out += shortcut8
        out = F.relu(out)
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

class one_net(nn.Module):
    def __init__(self, num_classes=10):
        super(one_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # block1
        self.block1_conv1 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.block1_bn1 = nn.BatchNorm2d(16)
        self.block1_conv2 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.block1_bn2 = nn.BatchNorm2d(16)
        self.block1_shortcut = nn.Sequential()
        # block2
        self.block2_conv1 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.block2_bn1 = nn.BatchNorm2d(16)
        self.block2_conv2 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.block2_bn2 = nn.BatchNorm2d(16)
        self.block2_shortcut = nn.Sequential()
        # block3
        self.block3_conv1 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.block3_bn1 = nn.BatchNorm2d(16)
        self.block3_conv2 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.block3_bn2 = nn.BatchNorm2d(16)
        self.block3_shortcut = nn.Sequential()
        # block4
        self.block4_conv1 = nn.Conv2d(16, 32, kernel_size=1, stride=2, padding=0, bias=False)
        self.block4_bn1 = nn.BatchNorm2d(32)
        self.block4_conv2 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.block4_bn2 = nn.BatchNorm2d(32)
        self.block4_shortcut = nn.Sequential(
                            nn.Conv2d(16, 32, kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(32)
                        )
        # block5
        self.block5_conv1 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.block5_bn1 = nn.BatchNorm2d(32)
        self.block5_conv2 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.block5_bn2 = nn.BatchNorm2d(32)
        self.block5_shortcut = nn.Sequential()
        # block6
        self.block6_conv1 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.block6_bn1 = nn.BatchNorm2d(32)
        self.block6_conv2 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.block6_bn2 = nn.BatchNorm2d(32)
        self.block6_shortcut = nn.Sequential()
        # block7
        self.block7_conv1 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0, bias=False)
        self.block7_bn1 = nn.BatchNorm2d(64)
        self.block7_conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.block7_bn2 = nn.BatchNorm2d(64)
        self.block7_shortcut = nn.Sequential(
                            nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(64)
                        )
        # block8
        self.block8_conv1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.block8_bn1 = nn.BatchNorm2d(64)
        self.block8_conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.block8_bn2 = nn.BatchNorm2d(64)
        self.block8_shortcut = nn.Sequential()
        # block9
        self.block9_conv1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.block9_bn1 = nn.BatchNorm2d(64)
        self.block9_conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.block9_bn2 = nn.BatchNorm2d(64)
        self.block9_shortcut = nn.Sequential()
        self.linear = nn.Linear(64, num_classes)
        self.plug1 = shuffle_plug_padding(32, 32, 16)
        # self.plug2 = shuffle_plug_padding(32, 32, 16)
        # self.plug3 = shuffle_plug_4(32, 32, 16)
        # self.plug4 = CycleTwist(32, 16, (1, 5))
        # self.plug5 = CycleTwist(32, 16, (1, 5))
        self.plug6 = nn.Sequential()
        self.plug7 = nn.Sequential()
        self.plug8 = nn.Sequential()
        self.plug9 = nn.Sequential()
        # self.plug2 = nn.Sequential()
        # self.plug0 = shuffle_plug_4(32, 32, 3)
        self.plug3 = nn.Sequential()
        self.plug4 = nn.Sequential()
        self.plug5 = nn.Sequential()
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
        # out = self.plug2(out)
        # block3
        shortcut3 = self.block3_shortcut(out)
        out = F.relu(self.block3_bn1(self.block3_conv1(out)))
        out = self.block3_bn2(self.block3_conv2(out))
        out += shortcut3
        out = F.relu(out)
        out = self.plug3(out)
        # block4
        # shortcut4 = self.block4_shortcut(out)
        shortcut4 = self.block4_shortcut(F.avg_pool2d(out, 2))
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
        shortcut7 = self.block7_shortcut(F.avg_pool2d(out, 2))
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


class ConvMixerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.convmixer = ConvMixer(32, 32, kernel_size=9, patch_size=2, n_classes=100)
    
    def forward(self, x):
        out = self.convmixer(x)
        return out

class ConvMixerShuffleNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding = nn.Conv2d(3, 32, 2, 2, 0)
        self.block1 = shuffle_block(16, 16, 32)
        self.block2 = shuffle_block(16, 16, 32)
        self.block3 = shuffle_block(16, 16, 32)
        self.block4 = shuffle_block(16, 16, 32)
        self.block5 = shuffle_block(16, 16, 32)
        self.downsample = BasicBlock(32, 64, 3, 2)
        self.block6 = shuffle_block(8, 8, 64)
        self.block7 = shuffle_block(8, 8, 64)
        self.block8 = shuffle_block(8, 8, 64)
        self.linear = nn.Linear(64, self.num_classes)
    
    def forward(self, x):
        out = self.embedding(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.downsample(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ConvMixerShuffleNet2(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.blocks = 4
        self.embedding = nn.Conv2d(3, 16, 3, 1, 1)
        self.stage1 = nn.Sequential(
            *[shuffle_block2(32, 32, 16) for _ in range(self.blocks)]
        )
        self.downsample = BasicBlock(16, 32, 3, 2)
        self.stage2 = nn.Sequential(
            *[shuffle_block2(16, 16, 32) for _ in range(self.blocks)]
        )
        self.downsample2 = BasicBlock(32, 64, 3, 2)
        self.stage3 = nn.Sequential(
            *[shuffle_block2(8, 8, 64) for _ in range(self.blocks)]
        )
        self.linear = nn.Linear(64, self.num_classes)
    
    def forward(self, x):
        out = self.embedding(x)
        out = self.stage1(out)
        out = self.downsample(out)
        out = self.stage2(out)
        out = self.downsample2(out)
        out = self.stage3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class NoPaddingNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.body = nn.Sequential(
            BasicBlock2(16, 16, 3, 1),
            BasicBlock2(16, 16, 3, 1),
            BasicBlock2(16, 32, 3, 1),
            BasicBlock2(32, 32, 3, 1),
            BasicBlock2(32, 64, 3, 1),
            BasicBlock2(64, 64, 3, 1),
            BasicBlock(64, 64, 3, 1)
        )
        self.linear = nn.Linear(64, self.num_classes)
    
    def forward(self, x):
        out = self.head(x)
        out = self.body(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class NormNet2(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding = nn.Conv2d(3, 16, 3, 1, 1)
        self.blocks = 9
        self.stage1 = nn.Sequential(
            *[normal_block2(16, 16) for _ in range(self.blocks)]
        )
        self.downsample = BasicBlock(16, 32, 3, 2)
        self.stage2 = nn.Sequential(
            *[normal_block2(32, 16) for _ in range(self.blocks)]
        )
        self.downsample2 = BasicBlock(32, 64, 3, 2)
        self.stage3 = nn.Sequential(
            *[normal_block2(64, 16) for _ in range(self.blocks)]
        )
        self.linear = nn.Linear(64, self.num_classes)
    
    def forward(self, x):
        out = self.embedding(x)
        out = self.stage1(out)
        out = self.downsample(out)
        out = self.stage2(out)
        out = self.downsample2(out)
        out = self.stage3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

if __name__ == "__main__":
    from torchsummary import summary
    model = model_trial_v1(num_classes=100)
    print(summary(model, (3, 32, 32), device='cpu'))



