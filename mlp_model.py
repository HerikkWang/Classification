from lib2to3.pgen2 import token
from statistics import mode
from ResNet20_raw import _weights_init
from component import *
import torch
import torch.nn as nn

class patch_merge(nn.Module):
    def __init__(self, C, W, H, out_channel) -> None:
        super().__init__()
        self.W = W
        self.H = H
        self.C = C
        self.linear = nn.Conv2d(4 * C, out_channel, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        out = x.reshape(-1, self.C, 2, self.W // 2, 2, self.H // 2)
        out = out.transpose(3, 4)
        out = out.reshape(-1, self.C * 2 * 2, self.W // 2, self.H // 2)
        out = self.gelu(self.bn(self.linear(out)))
        return out

class sMLP(nn.Module):
    def __init__(self, channel, W, H) -> None:
        super().__init__()
        self.proj_h = nn.Conv2d(H, H, 1)
        self.proj_w = nn.Conv2d(W, W, 1)
        self.fuse = nn.Conv2d(3 * channel, channel, 1)
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm2d(channel)
    
    def forward(self, x):
        x_h = self.proj_h(x.permute(position_coding("NCWH", "NHCW"))).permute(position_coding("NHCW", "NCWH"))
        x_w = self.proj_w(x.permute(position_coding("NCWH", "NWHC"))).permute(position_coding("NWHC", "NCWH"))
        x_id = x
        x_fuse = torch.cat([x_h, x_w, x_id],dim=1)
        out = self.gelu(self.bn(self.fuse(x_fuse)))
        return out

class token_mixing(nn.Module):
    def __init__(self, channel, W, H) -> None:
        super().__init__()
        self.dw_conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, groups=channel)
        self.gelu1 = nn.GELU()
        self.bn = nn.BatchNorm2d(channel)
        self.sMLP = sMLP(channel, W, H)
    
    def forward(self, x):
        out = self.gelu1(self.bn(self.dw_conv(x)))
        out = self.sMLP(out)
        return out

class sMLP_block(nn.Module):
    def __init__(self, in_channel, out_channel, W, H, N, merge=False) -> None:
        super().__init__()
        if merge:
            self.patch_merge = patch_merge(in_channel, W * 2, H * 2, out_channel=out_channel)
        else:
            self.patch_merge = nn.Sequential()
        self.token_mixing = nn.Sequential(
            self.patch_merge,
            *[token_mixing(out_channel, W, H) for _ in range(N)],
        )
    
    def forward(self, x):
        out = self.token_mixing(x)
        return out

class SparseMLP(nn.Module):
    def __init__(self, W=32, H=32, blocks=[2, 4, 4, 2], num_classes=100) -> None:
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, stride=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )
        self.block1 = sMLP_block(16, 16, W, H, blocks[0], merge=False)
        self.block2 = sMLP_block(16, 32, W // 2, H // 2, blocks[1], merge=True)
        self.block3 = sMLP_block(32, 64, W // 4, H // 4, blocks[2], merge=True)
        self.block4 = sMLP_block(64, 128, W // 8, H // 8, blocks[3], merge=True)
        self.linear = nn.Linear(128, num_classes)
        self.apply(_weights_init)
    
    def forward(self, x):
        out = self.embedding(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

if __name__ == "__main__":
    from torchsummary import summary
    model = SparseMLP(32, 32, blocks=[2, 4, 4, 2])
    print(model)
    print(summary(model, (3, 32, 32), device='cpu'))

        




        