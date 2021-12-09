import torch
import torch.nn as nn
import torch.nn.functional as F
from .box_ops import giou_score
from .create_labels import label_creator


class FocalWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean', gamma=2.0, alpha=0.25):
        super(FocalWithLogitsLoss, self).__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets, mask=None):
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(input=logits, 
                                                     target=targets, 
                                                     reduction="none"
                                                     )
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        loss = ce_loss * ((1.0 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            pos_inds = (targets == 1.0).float()
            # [B, HW, KA, C] -> [B,]
            num_pos = pos_inds.sum([1, 2, 3]).clamp(1.0)

            if mask is None:
                # [B, HW, KA, C] -> [B,]
                loss = loss.sum([1, 2, 3]) / num_pos
                loss = loss.sum()
            else:
                # [B, HW,] -> [B, HW, 1, 1]
                mask = mask[..., None, None]
                loss = (loss * mask).sum([1, 2, 3]) / num_pos
                loss = loss.sum()

        elif self.reduction == "sum":
            if mask is None:
                # [B, HW, KA, C] -> [B,]
                loss = loss.sum()
            else:
                # [B, HW,] -> [B, HW, 1, 1]
                mask = mask[..., None, None]
                loss = (loss * mask).sum()

        return loss


class SetCriterion(nn.Module):
    def __init__(self, cfg, loss_cls_weight=1.0, loss_reg_weight=1.0, num_classes=80):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.loss_cls_weight = loss_cls_weight
        self.loss_reg_weight = loss_reg_weight

        self.cls_loss_f = FocalWithLogitsLoss(reduction='mean')

    def loss_labels(self, pred_cls, target, mask=None):
        # groundtruth    
        target_labels = target[..., :self.num_classes].float() # [B, HW, KA, C]

        # cls loss
        loss_cls = self.cls_loss_f(pred_cls, target_labels, mask)

        return loss_cls

    def loss_bboxes(self, pred_box, target, mask=None):
        # groundtruth    
        target_bboxes = target[..., self.num_classes:self.num_classes+4] # [B, HW, KA, 4]
        target_pos = target[..., -1].float()                             # [B, HW, KA,]
        num_pos = target_pos.sum([1, 2]).clamp(1.0)                      # [B,]

        # reg loss
        B, HW, KA, _ = pred_box.size()
        # decode bbox: [B, HW, KA, 4] -> [B x HW x KA, 4]
        x1y1x2y2_pred = pred_box.view(-1, 4)
        x1y1x2y2_gt = target_bboxes.view(-1, 4)

        # giou: [B x HW x KA,]
        pred_giou = giou_score(x1y1x2y2_pred, x1y1x2y2_gt)
        # [B x HW x KA,] -> [B, HW, KA,]
        pred_giou = pred_giou.view(B, HW, KA)
        loss_giou = 1. - pred_giou
        if mask is None:
            loss_reg = (loss_giou * target_pos).sum([1, 2]) / num_pos
            loss_reg = loss_reg.sum()
        else:
            # [B, HW,] -> [B, HW, 1]
            mask = mask[..., None]
            loss_reg = (loss_giou * mask * target_pos).sum([1, 2]) / num_pos
            loss_reg = loss_reg.sum()
        
        return loss_reg


    def forward(self, anchor_boxes, outputs, targets):
        """
            outputs["pred_cls"]: (tensor) [B, HW, KA, C]
            outputs["pred_giou"]: (tensor) [B, HW, KA, 1]
            outputs["mask"]: (tensor) [B, HW]
            target: (tensor) [B, HW, KA, C+4+1]
        """
        # make labels
        targets = label_creator(targets=targets, 
                                anchor_boxes=anchor_boxes, 
                                num_classes=self.num_classes,
                                topk=self.cfg['topk'],
                                igt=self.cfg['ignore_thresh'])

        batch_size = outputs["pred_cls"].size(0)
        # compute class loss
        loss_labels = self.loss_labels(outputs["pred_cls"], targets, outputs["mask"])
        loss_labels /= batch_size
        # compute bboxes loss
        loss_bboxes = self.loss_bboxes(outputs["pred_box"], targets, outputs["mask"])
        loss_bboxes /= batch_size

        # total loss
        losses = self.loss_cls_weight * loss_labels + self.loss_reg_weight * loss_bboxes

        return loss_labels, loss_bboxes, losses


def build_criterion(args, cfg, num_classes=80):
    criterion = SetCriterion(cfg=cfg,
                             loss_cls_weight=args.loss_cls_weight,
                             loss_reg_weight=args.loss_reg_weight,
                             num_classes=num_classes)
    return criterion

    
if __name__ == "__main__":
    pass
