import torch
import torch.nn as nn
import torch.nn.functional as F
from .box_ops import giou_score
from .create_labels import label_creator
from utils.vis import vis_targets


class FocalWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean', gamma=2.0, alpha=0.25):
        super(FocalWithLogitsLoss, self).__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets, targets_valid, mask=None):
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(input=logits, 
                                                     target=targets, 
                                                     reduction="none"
                                                     )
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        loss = ce_loss * ((1.0 - p_t) ** self.gamma)
        # valid loss. Here we ignore the loss of ignore samples
        loss = loss * targets_valid[..., None]
        # valid loss. Here we ignore the loss of samples who are not in the image if mask is not None.
        loss = loss if mask is None else loss * mask[..., None, None]

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            pos_inds = (targets == 1.0).float()
            # [B, HW, KA, C] -> [B,]
            num_pos = pos_inds.sum([1, 2, 3]).clamp(1.0)

            # [B, HW, KA, C] -> [B,]
            loss = loss.sum([1, 2, 3]) / num_pos
            loss = loss.sum()

        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class Criterion(nn.Module):
    def __init__(self, cfg, device, loss_cls_weight=1.0, loss_reg_weight=1.0, num_classes=80):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        self.loss_cls_weight = loss_cls_weight
        self.loss_reg_weight = loss_reg_weight

        self.cls_loss_f = FocalWithLogitsLoss(reduction='mean')


    def loss_labels(self, pred_cls, target, mask=None):
        # groundtruth    
        target_labels = target[..., :self.num_classes].float() # [B, HW, KA, C]
        target_valid = (target[..., -1] > -1.0).float()        # [B, HW, KA,]

        # cls loss
        loss_cls = self.cls_loss_f(pred_cls, target_labels, target_valid, mask)

        return loss_cls


    def loss_bboxes(self, pred_box, target, mask=None):
        # groundtruth    
        target_bboxes = target[..., self.num_classes:self.num_classes+4] # [B, HW, KA, 4]
        target_pos = (target[..., -1] > 0.).float()                      # [B, HW, KA,]
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
        loss_reg = 1. - pred_giou if mask is None else (1. - pred_giou) * mask[..., None]

        loss_reg = (loss_reg * target_pos).sum([1, 2]) / num_pos
        loss_reg = loss_reg.sum()
        
        return loss_reg


    def forward(self, anchor_boxes, outputs, targets, stride=32, images=None, vis_labels=False):
        """
            anchor_boxes: (tensor) [1, HW, KA, 4]
            outputs["pred_cls"]: (tensor) [B, HW, KA, C]
            outputs["pred_giou"]: (tensor) [B, HW, KA, 1]
            outputs["mask"]: (tensor) [B, HW]
            target: (list) a list of annotations
            images: (tensor) [B, 3, H, W]
            vis_labels: (bool) visualize labels to check positive samples
        """
        batch_size = outputs["pred_cls"].size(0)
        # make labels
        targets = label_creator(targets=targets, 
                                stride=stride,
                                anchor_boxes=anchor_boxes, 
                                num_classes=self.num_classes,
                                topk=self.cfg['topk'],
                                iou_t=self.cfg['iou_thresh'],
                                igt=self.cfg['ignore_thresh'])
        # [B, HW, KA, C+4+1]
        targets = targets.to(self.device)

        # vis labels
        if vis_labels:
            vis_targets(images, targets, anchor_boxes)

        # compute class loss
        loss_labels = self.loss_labels(outputs["pred_cls"], targets, outputs["mask"])
        loss_labels /= batch_size

        # compute bboxes loss
        loss_bboxes = self.loss_bboxes(outputs["pred_box"], targets, outputs["mask"])
        loss_bboxes /= batch_size

        # total loss
        losses = self.loss_cls_weight * loss_labels + self.loss_reg_weight * loss_bboxes

        return loss_labels, loss_bboxes, losses


def build_criterion(args, cfg, device, num_classes=80):
    criterion = Criterion(cfg=cfg,
                          device=device,
                          loss_cls_weight=args.loss_cls_weight,
                          loss_reg_weight=args.loss_reg_weight,
                          num_classes=num_classes)
    return criterion

    
if __name__ == "__main__":
    pass
