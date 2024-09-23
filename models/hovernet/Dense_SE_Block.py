import torch
import torch.nn as nn
from .utils import crop_op, crop_to_shape
from collections import OrderedDict


class Net(nn.Module):
    """ A base class provides a common weight initialisation scheme."""

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__

            # ! Fixed the type checking
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            if "norm" in classname.lower():
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            if "linear" in classname.lower():
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return x



class DeepWise_SE(nn.Module):
    def __init__(self, in_channel):
        super(DeepWise_SE, self).__init__()
        self.conv_deepWise = nn.Conv2d(in_channel, in_channel//16, kernel_size=3, padding=1, stride=1, groups=in_channel)
        self.conv_dot = nn.Conv2d(in_channel//16, in_channel, kernel_size=1, padding=0, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv_deepWise(x)
        x = self.relu(x)
        x = self.conv_dot(x)
        x = self.sigmoid(x)
        return x


class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        # channel attention module
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channel, in_channel//16, kernel_size=1, stride=1, padding=0, groups=in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel//16, in_channel, kernel_size=1, stride=1, padding=0),
        )

        # spatial attention module
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=7 // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.maxpool(x)
        max_out = self.mlp(max_out)
        avg_out = self.avgpool(x)
        avg_out = self.mlp(avg_out)
        channel_out = self.sigmoid(max_out + avg_out)
        x = x * channel_out

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = x * spatial_out
        return x









class SE_Block(nn.Module):
    def __init__(self, in_ch):
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_ch, in_ch//16, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_ch//16, in_ch, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.sigmoid(x)
        return out


class DenseBlock(Net):
    """Dense Block as defined in:

    Huang, Gao, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger.
    "Densely connected convolutional networks." In Proceedings of the IEEE conference
    on computer vision and pattern recognition, pp. 4700-4708. 2017.

    Only performs `valid` convolution.

    """

    def __init__(self, in_ch, unit_ksize, unit_ch, unit_count, split=1):
        super(DenseBlock, self).__init__()
        assert len(unit_ksize) == len(unit_ch), "Unbalance Unit Info"

        self.nr_unit = unit_count  # 8 / 4
        self.in_ch = in_ch  # 256 / 128
        self.unit_ch = unit_ch  # [128， 32]

        # ! For inference only so init values for batchnorm may not match tensorflow
        unit_in_ch = in_ch
        self.units = nn.ModuleList()
        for idx in range(unit_count):
            self.units.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("preact_bna/bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                            ("preact_bna/relu", nn.ReLU(inplace=True)),
                            (
                                "conv1",
                                nn.Conv2d(
                                    unit_in_ch,
                                    unit_ch[0],
                                    unit_ksize[0],
                                    stride=1,
                                    padding=0,
                                    bias=False,
                                ),
                            ),
                            ("conv1/bn", nn.BatchNorm2d(unit_ch[0], eps=1e-5)),
                            ("conv1/relu", nn.ReLU(inplace=True)),
                            # ('conv2/pool', TFSamepaddingLayer(ksize=unit_ksize[1], stride=1)),
                            (
                                "conv2",
                                nn.Conv2d(
                                    unit_ch[0],
                                    unit_ch[1],
                                    unit_ksize[1],
                                    groups=split,
                                    stride=1,
                                    padding=0,
                                    bias=False,
                                ),
                            ),

                        ]
                    )
                )
            )
            unit_in_ch += unit_ch[1]
        self.SE = SE_Block(unit_in_ch)
        self.d_SE = DeepWise_SE(unit_in_ch)
        self.blk_bna = nn.Sequential(
            OrderedDict(
                [
                    ("bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )

    def out_ch(self):
        return self.in_ch + self.nr_unit * self.unit_ch[-1]

    def forward(self, prev_feat):
        for idx in range(self.nr_unit):
            new_feat = self.units[idx](prev_feat)
            prev_feat = crop_to_shape(prev_feat, new_feat)  # 将prev_feat 裁剪成 new_feat 的大小
            prev_feat = torch.cat([prev_feat, new_feat], dim=1)
        # out = self.SE(prev_feat)
        out = self.d_SE(prev_feat)
        prev_feat = prev_feat * out
        prev_feat = self.blk_bna(prev_feat)
        return prev_feat