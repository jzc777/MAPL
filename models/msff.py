import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .coordatt import CoordAtt


class MSFFBlock(nn.Module):
    def __init__(self, in_channel):
        super(MSFFBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channel // 2, in_channel // 2, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x_conv = self.conv1(x)

        x = self.conv2(x)
        return x

class MSFF(nn.Module):
    def __init__(self):
        super(MSFF, self).__init__()
        self.blk1 = MSFFBlock(128)
        self.blk2 = MSFFBlock(256)
        self.blk3 = MSFFBlock(512)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.upconv32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        )
        self.upconv21 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, features):
        # features = [level1, level2, level3]
        f1, f2, f3 = features

        # MSFF Module
        f1_k = self.blk1(f1)
        f2_k = self.blk2(f2)
        f3_k = self.blk3(f3)

        # 上采样和特征融合
        f2_f = f2_k + self.upconv32(f3_k)
        f1_f =f1_k + self.upconv21(f2_f)
        return [f1_f, f2_f, f3_k]

        # 以前的代码中使用了空间注意