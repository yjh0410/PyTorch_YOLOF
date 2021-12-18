import torch
import torch.nn as nn
from .conv import Conv
from utils import weight_init


# Dilated Encoder
class Bottleneck(nn.Module):
    def __init__(self, c, d=1, e=0.5, norm='BN', act=True):
        super(Bottleneck, self).__init__()
        c_ = int(c * e)
        self.branch = nn.Sequential(
            Conv(c, c_, k=1, norm=norm, act=act),
            Conv(c_, c_, k=3, norm=norm, p=d, d=d, act=act),
            Conv(c_, c, k=1, norm=norm, act=act)
        )

    def forward(self, x):
        return x + self.branch(x)


class DilatedEncoder(nn.Module):
    """ DilateEncoder """
    def __init__(self, c1, c2, e=0.5, norm='BN', act=True, dilation_list=[2, 4, 6, 8]):
        super(DilatedEncoder, self).__init__()
        self.projector = nn.Sequential(
            Conv(c1, c2, k=1, norm=norm, act=False),
            Conv(c2, c2, k=3, p=1, norm=norm, act=False)
        )
        encoders = []
        for d in dilation_list:
            encoders.append(Bottleneck(c=c2, d=d, e=e, norm=norm, act=act))
        self.encoders = nn.Sequential(*encoders)

        self._init_weight()

    def _init_weight(self):
        for m in self.projector:
            if isinstance(m, nn.Conv2d):
                weight_init.c2_xavier_fill(m)
                weight_init.c2_xavier_fill(m)
            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.encoders.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.projector(x)
        x = self.encoders(x)

        return x
