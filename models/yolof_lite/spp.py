import torch
import torch.nn as nn

from ..basic.conv import Conv


# Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
class SPPF(nn.Module):
    def __init__(self, in_dim, out_dim, k=5, num_maxpool=3, act_type='relu', norm_type='BN'):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        inter_dim = in_dim // 2  # hidden channels
        self.num_maxpool = num_maxpool
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(inter_dim * (num_maxpool+1), out_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y = self.m(x)
        ys = [y]
        for _ in range(self.num_maxpool - 1):
            y = self.m(y)
            ys.append(y)

        return self.cv2(torch.cat((x, *ys), 1))


# SPP block with CSP module
class SPPBlockCSP(nn.Module):
    """
        CSP Spatial Pyramid Pooling Block
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.5,
                 kernel_size=5,
                 num_maxpool=3, 
                 act_type='relu',
                 norm_type='BN'
                 ):
        super(SPPBlockCSP, self).__init__()
        inter_dim = int(out_dim * expand_ratio)
        # input proj layer
        self.projector = nn.Sequential(
            Conv(in_dim, out_dim, k=1, act_type=None, norm_type=norm_type),
            Conv(out_dim, out_dim, k=3, p=1, act_type=None, norm_type=norm_type)
        )
        # spp layers
        self.cv1 = Conv(out_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(out_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.m = nn.Sequential(
            Conv(inter_dim, inter_dim, k=3, p=1, 
                 act_type=act_type, norm_type=norm_type),
            SPPF(inter_dim, inter_dim,
                 k=kernel_size,
                 num_maxpool=num_maxpool,
                 act_type=act_type, 
                 norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, p=1, 
                 act_type=act_type, norm_type=norm_type)
        )
        self.cv3 = Conv(inter_dim * 2, out_dim, k=1, act_type=act_type, norm_type=norm_type)

        
    def forward(self, x):
        x = self.projector(x)
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m(x2)
        y = self.cv3(torch.cat([x1, x3], dim=1))

        return y



def build_spp(cfg, in_dim, out_dim):
    spp = SPPBlockCSP(
        in_dim, out_dim,
        expand_ratio=cfg['expand_ratio'],
        kernel_size=cfg['kernel_size'],
        num_maxpool=cfg['num_maxpool'],
        act_type=cfg['neck_act'],
        norm_type=cfg['neck_norm'])

    return spp
