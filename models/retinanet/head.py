import torch
import torch.nn as nn

from ..basic.conv import Conv


class DecoupledHead(nn.Module):
    def __init__(self, 
                 head_dim=256,
                 num_classes=80,
                 num_cls_heads=4,
                 num_reg_heads=4,
                 num_anchors=9,
                 act_type='relu',
                 norm_type='BN'):
        super().__init__()

        print('==============================')
        print('Head: Decoupled Head')
        if norm_type is None:
            bias = True
        else:
            bias = False

        self.cls_feats = nn.Sequential(*[Conv(head_dim, 
                                              head_dim, 
                                              k=3, p=1, s=1, 
                                              act_type=act_type, 
                                              norm_type=norm_type,
                                              bias=bias) for _ in range(num_cls_heads)])
        self.reg_feats = nn.Sequential(*[Conv(head_dim, 
                                              head_dim, 
                                              k=3, p=1, s=1, 
                                              act_type=act_type, 
                                              norm_type=norm_type,
                                              bias=bias) for _ in range(num_reg_heads)])

        # pred
        self.cls_pred = nn.Conv2d(head_dim, num_classes*num_anchors,  kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(head_dim, 4*num_anchors, kernel_size=3, padding=1)

        self._init_weight()


    def _init_weight(self):
        # init weight of detection head
        for m in [self.cls_feats, self.reg_feats]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # init cls pred
        nn.init.normal_(self.cls_pred.weight, mean=0, std=0.01)
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.cls_pred.bias, bias_value)
        # init reg pred
        nn.init.normal_(self.reg_pred.weight, mean=0, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 0.0)


    def forward(self, x):
        """
            in_feats: (Tensor) [B, C, H, W]
        """
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)

        cls_pred = self.cls_pred(cls_feats)
        reg_pred = self.reg_pred(reg_feats)

        return cls_pred, reg_pred


# build head
def build_head(cfg, num_classes, num_anchors):
    head = DecoupledHead(
        head_dim=cfg['head_dim'],
        num_classes=num_classes,
        num_anchors=num_anchors,
        num_cls_heads=cfg['num_cls_heads'],
        num_reg_heads=cfg['num_reg_heads'],
        act_type=cfg['head_act_type'],
        norm_type=cfg['head_norm_type']
        )

    return head
    