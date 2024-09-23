import torch
import torch.nn as nn


class Residual_Inception_Block(nn.Module):
    def __init__(self, in_ch, out_ch, unit_count, stride=1):
        super(Residual_Inception_Block, self).__init__()
        # assert len(unit_ksize) == len(unit_ch), "Unbalance Unit Info, please check your unit_ksize and unit_ch"

        self.nr_unit = unit_count  # 这个block本次循环几次？
        self.in_ch = in_ch  # block 的 in_channel
        self.out_ch = out_ch  # 在该次block的output_channel分别是什么

        unit_in_ch = in_ch
        self.units = nn.ModuleList()
        for idx in range(unit_count):
            # unit_layer = [
            #     ("preact/bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
            #     ("preact/relu", nn.ReLU(inplace=True)),
            # ]
            preact = nn.Sequential(
                nn.BatchNorm2d(unit_in_ch, eps=1e-5),
                nn.ReLU(inplace=True),
            )
            # unit_layer_3x3 = [
            #     ("conv3x3",
            #      nn.Conv2d(unit_in_ch, in_ch * 2, kernel_size=3, stride=1, padding=1, bias=False)),
            #     ("conv3x3/bn", nn.BatchNorm2d(in_ch * 2, eps=1e-5)),
            #     ("conv3x3/relu", nn.ReLU(inplace=True)),
            # ]
            conv3x3 = nn.Sequential(
                nn.Conv2d(unit_in_ch, in_ch * 2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(in_ch * 2, eps=1e-5),
                nn.ReLU(inplace=True),
            )
            # unit_layer_5x5 = [
            #     ("conv5x5",
            #      nn.Conv2d(unit_in_ch, in_ch, kernel_size=5, stride=1, padding=2, bias=False)),
            #     ("conv1x1", nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, stride=1, padding=0, bias=False)),
            #     ("conv5x5/bn", nn.BatchNorm2d(in_ch * 2, eps=1e-5)),
            #     ("conv5x5/relu", nn.ReLU(inplace=True)),
            # ]
            conv5x5 = nn.Sequential(
                nn.Conv2d(unit_in_ch, in_ch, kernel_size=5, stride=1, padding=2, bias=False),
                nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(in_ch * 2, eps=1e-5),
                nn.ReLU(inplace=True),
            )
            # unit_layer_end = [
            #     ("conv/end", nn.Conv2d(in_ch * 4, out_ch, kernel_size=3,
            #                            stride=stride if idx == 0 else 1, bias=False)),
            # ]
            if idx == 0:
                conv_end = nn.Conv2d(in_ch * 4, out_ch, kernel_size=3, stride=stride, bias=False)
            else:
                conv_end = nn.Conv2d(in_ch * 4, out_ch, kernel_size=3, stride=1, bias=False)
            unit = nn.ModuleDict({
                'preact': preact,
                'conv3x3': conv3x3,
                'conv5x5': conv5x5,
                'conv_end': conv_end,
            })
            self.units.append(unit)
            unit_in_ch = out_ch

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = None

        self.blk_bna = nn.Sequential(
            nn.BatchNorm2d(out_ch, eps=1e-5),
            nn.ReLU(inplace=True),
        )

    def forward(self, prev_fast, freeze=False):
        if self.shortcut is None:
            shortcut = prev_fast
        else:
            shortcut = self.shortcut(prev_fast)
        for unit in self.units:
            x_preact = unit['preact'](prev_fast)
            x_3 = unit['conv3x3'](x_preact)
            x_5 = unit['conv5x5'](x_preact)
            x_cat = torch.cat((x_3, x_5), dim=1)
            out = unit['conv_end'](x_cat)
            shortcut = shortcut + out
        out = self.blk_bna(shortcut)
        return out



