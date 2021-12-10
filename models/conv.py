import torch.nn as nn
from torch.nn.functional import group_norm


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, p=0, s=1, d=1, g=1, norm='BN', act=True, bias=False):
        super(Conv, self).__init__()
        if act:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias),
                nn.BatchNorm2d(c2) if norm == 'BN' else nn.GroupNorm(c2, num_groups=32),
                nn.ReLU(inplace=True)
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias),
                nn.BatchNorm2d(c2) if norm == 'BN' else nn.GroupNorm(c2, num_groups=32)
            )

    def forward(self, x):
        return self.convs(x)
