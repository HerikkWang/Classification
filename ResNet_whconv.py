import torch.nn as nn
import torch.nn.functional as F
from utils.functions import position_coding, _weights_init

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, feature_size, kernel_size, stride=1, if_shuffle=False, ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(feature_size // stride, feature_size // stride, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.if_shuffle = if_shuffle
        # self.bn3 = nn.BatchNorm2d(in_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        if self.if_shuffle:
            out = x.permute(position_coding("NCWH", "NHCW"))
            out = self.conv1(out)
            out = out.permute(position_coding("NHCW", "NCWH"))
        else:
            out = x
        out = F.relu(self.bn1(self.conv3(out)))
        if self.if_shuffle:
            out = out.permute(position_coding("NCWH", "NWHC"))
            out = self.conv2(out)
            out = out.permute(position_coding("NWHC", "NCWH"))
        # out = self.bn3(out)
        out = self.bn2(self.conv4(out))
        # out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResWHNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.block1 = BasicBlock(16, 16, 32, 3, 1, True)
        self.block2 = BasicBlock(16, 16, 32, 3, 1, True)
        self.block3 = BasicBlock(16, 16, 32, 3, 1, True)
        self.block4 = BasicBlock(16, 32, 32, 3, 2)
        self.block5 = BasicBlock(32, 32, 16, 3, 1)
        self.block6 = BasicBlock(32, 32, 16, 3, 1)
        self.block7 = BasicBlock(32, 64, 16, 3, 2)
        self.block8 = BasicBlock(64, 64, 8, 3, 1)
        self.block9 = BasicBlock(64, 64, 8, 3, 1)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
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

if __name__ == "__main__":
    from torchsummary import summary
    model = ResWHNet(num_classes=100)
    print(summary(model, (3, 32, 32), device='cpu'))



