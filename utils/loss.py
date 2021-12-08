import torch
import torch.nn as nn
import torch.nn.functional as F
from .box_ops import giou_score


class FocalWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean', gamma=2.0, alpha=0.25):
        super(FocalWithLogitsLoss, self).__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
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
            batch_size = logits.size(0)
            pos_inds = (targets == 1.0).float()
            # [B, H*W, C] -> [B,]
            num_pos = pos_inds.sum([1, 2]).clamp(1)
            loss = loss.sum([1, 2])
    
            loss = (loss / num_pos).sum() / batch_size

        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss


class SetCriterion(nn.Module):
    def __init__(self, loss_cls_weight=1.0, loss_reg_weight=1.0, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.loss_cls_weight = loss_cls_weight
        self.loss_reg_weight = loss_reg_weight

        self.cls_loss_f = FocalWithLogitsLoss(reduction='mean')

    def loss_labels(self, pred_cls, target):
        # groundtruth    
        target_labels = target[..., :self.num_classes].float() # [B, HW, KA, C]

        # cls loss
        cls_loss = self.cls_loss_f(pred_cls, target_labels)

        return cls_loss

    def loss_bboxes(self, pred_box, target):
        # groundtruth    
        target_bboxes = target[..., self.num_classes:self.num_classes+4] # [B, HW, KA, 4]
        target_pos = target[..., -1].float()                             # [B, HW, KA,]
        num_pos = target_pos.sum(-1, keepdim=True).clamp(1)              # [B, HW x KA,]

        # reg loss
        # TODO: cocmpute GIoU
        pred_giou = None
        reg_loss = ((1. - pred_giou) * target_pos / num_pos).sum()

    def forward(self, pred_cls, pred_box, target):
        """
            pred_cls: (tensor) [B, HW, KA, C]
            pred_giou: (tensor) [B, HW, KA, 1]
            target: (tensor) [B, HW, KA, C+4+1]
        """
        batch_size = pred_cls.size(0)
        # compute class loss
        loss_labels = self.loss_labels(pred_cls, target)
        # compute bboxes loss
        loss_bboxes = self.loss_bboxes(pred_box, target)

        # total loss
        losses = self.loss_cls_weight * loss_labels + self.loss_reg_weight * loss_bboxes
        losses /= batch_size

        return losses


def build_criterion(args, num_classes=80):
    criterion = SetCriterion(loss_cls_weight=args.loss_cls_weight,
                             loss_reg_weight=args.loss_reg_weight,
                             num_classes=num_classes)
    return criterion

    
if __name__ == "__main__":
    pass
