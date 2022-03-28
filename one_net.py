import torch.nn as nn
import torch.nn.functional as F
from utils.functions import position_coding, _weights_init
from component import *
from options import parser


class basic_stage(nn.Module):
    def __init__(self, block_num, kernel_size, in_ch, out_ch, if_plug=False, fm_size=None) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.block_num = block_num
        self.kernel_size = kernel_size
        self.stage = nn.Sequential()
        self.layers = self._make_layer(if_plug=if_plug, fm_size=fm_size)
    
    def _make_layer(self, if_plug, fm_size):
        if if_plug:
            if self.in_ch == self.out_ch:
                return nn.Sequential(
                    shuffle_plug_3(fm_size, fm_size, self.in_ch),
                    *[BasicBlock(self.in_ch, self.out_ch, self.kernel_size) for i in range(self.block_num)])
            else:
                return nn.Sequential(
                    shuffle_plug_3(fm_size, fm_size, self.in_ch),
                    BasicBlock(self.in_ch, self.out_ch, self.kernel_size, stride=2),
                    nn.Sequential(*[BasicBlock(self.out_ch, self.out_ch, self.kernel_size) for i in range(self.block_num - 1)])
                )
        else:
            if self.in_ch == self.out_ch:
                return nn.Sequential(*[BasicBlock(self.in_ch, self.out_ch, self.kernel_size) for i in range(self.block_num)])
            else:
                return nn.Sequential(
                    BasicBlock(self.in_ch, self.out_ch, self.kernel_size, stride=2),
                    nn.Sequential(*[BasicBlock(self.out_ch, self.out_ch, self.kernel_size) for i in range(self.block_num - 1)])
                )

    def forward(self, x):
        return self.layers(x)

class oneconv_net(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.channel = 16
        self.blocks_per_stage = args.blocks_per_stage
        self.kernel_size = args.kernel_size
        self.stages = args.stages
        self.ch_in_stages = [self.channel] + [self.channel * (2 ** i) for i in range(self.stages)]
        # print(self.ch_in_stages)
        self.num_classes = args.num_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.body = nn.Sequential(
            *[basic_stage(self.blocks_per_stage, self.kernel_size, self.ch_in_stages[i], self.ch_in_stages[i + 1], False, 32 // (2 ** ((i - 1) if i >= 1 else 0))) for i in range(self.stages)]
        )
        # print(self.stage_layer)
        self.linear = nn.Linear(64, self.num_classes)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.body(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

if __name__ == "__main__":
    args = parser.parse_args()
    from torchsummary import summary
    model = oneconv_net(args)
    print(summary(model, (3, 32, 32), device='cpu'))


