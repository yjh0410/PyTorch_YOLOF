import torch.nn as nn


def get_activation(act_type=None):
    if act_type is None:
        return nn.Identity()
    elif act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'mish':
        return nn.Mish(inplace=True)
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)


def get_norm(in_dim, norm_type=None):
    if norm_type is None:
        return nn.Identity()
    elif norm_type == 'BN':
        return nn.BatchNorm2d(in_dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(32, in_dim)
    elif norm_type == 'IN':
        return nn.InstanceNorm2d(in_dim)

   
# Basic conv layer
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, p=0, s=1, d=1, g=1, act_type='relu', norm_type='BN', depthwise=False, bias=False):
        super(Conv, self).__init__()
        if depthwise:
            assert c1 == c2
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=c1, bias=bias),
                get_norm(c2, norm_type),
                get_activation(act_type),
                nn.Conv2d(c2, c2, kernel_size=1, bias=bias),
                get_norm(c2, norm_type),
                get_activation(act_type)
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias),
                get_norm(c2, norm_type),
                get_activation(act_type)
            )


    def forward(self, x):
        return self.convs(x)
