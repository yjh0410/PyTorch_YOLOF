import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import Matcher, OTA_Matcher
from utils.box_ops import *
from utils.misc import sigmoid_focal_loss
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized


class Criterion(object):
    def __init__(self, cfg, device, num_classes=80):
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        self.alpha = cfg['alpha']
        self.gamma = cfg['gamma']
        self.loss_cls_weight = cfg['loss_cls_weight']
        self.loss_reg_weight = cfg['loss_reg_weight']
        self.loss_ctn_weight = cfg['loss_ctn_weight']
        if cfg['matcher'] == 'matcher':
            self.matcher = Matcher(cfg,
                                   num_classes=num_classes,
                                   box_weights=[1., 1., 1., 1.])
        elif cfg['matcher'] == 'ota_matcher':
            self.matcher = OTA_Matcher(cfg, 
                                       num_classes, 
                                       box_weights=[1., 1., 1., 1.])


    def loss_labels(self, pred_cls, tgt_cls, num_boxes=1.0):
        """
            pred_cls: (Tensor) [N, C]
            tgt_cls:  (Tensor) [N, C]
        """
        # cls loss: [V, C]
        loss_cls = sigmoid_focal_loss(pred_cls, tgt_cls, self.alpha, self.gamma, reduction='none')

        return loss_cls.sum() / num_boxes


    def loss_bboxes(self, pred_delta, tgt_delta, bbox_quality=None, num_boxes=1.0):
        """
            pred_box: (Tensor) [N, 4]
            tgt_box:  (Tensor) [N, 4]
        """
        pred_delta = torch.cat((-pred_delta[..., :2], pred_delta[..., 2:]), dim=-1)
        tgt_delta = torch.cat((-tgt_delta[..., :2], tgt_delta[..., 2:]), dim=-1)

        eps = torch.finfo(torch.float32).eps

        pred_area = (pred_delta[..., 2] - pred_delta[..., 0]).clamp_(min=0) \
            * (pred_delta[..., 3] - pred_delta[..., 1]).clamp_(min=0)
        tgt_area = (tgt_delta[..., 2] - tgt_delta[..., 0]).clamp_(min=0) \
            * (tgt_delta[..., 3] - tgt_delta[..., 1]).clamp_(min=0)

        w_intersect = (torch.min(pred_delta[..., 2], tgt_delta[..., 2])
                    - torch.max(pred_delta[..., 0], tgt_delta[..., 0])).clamp_(min=0)
        h_intersect = (torch.min(pred_delta[..., 3], tgt_delta[..., 3])
                    - torch.max(pred_delta[..., 1], tgt_delta[..., 1])).clamp_(min=0)

        area_intersect = w_intersect * h_intersect
        area_union = tgt_area + pred_area - area_intersect
        ious = area_intersect / area_union.clamp(min=eps)

       # giou
        g_w_intersect = torch.max(pred_delta[..., 2], tgt_delta[..., 2]) \
            - torch.min(pred_delta[..., 0], tgt_delta[..., 0])
        g_h_intersect = torch.max(pred_delta[..., 3], tgt_delta[..., 3]) \
            - torch.min(pred_delta[..., 1], tgt_delta[..., 1])
        ac_uion = g_w_intersect * g_h_intersect
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
        loss_box = 1 - gious

        if bbox_quality is not None:
            loss_box = loss_box * bbox_quality.view(loss_box.size())

        return loss_box.sum() / num_boxes


    # The origin loss of FCOS
    def basic_losses(self, outputs, targets):
        """
            outputs['pred_cls']: (Tensor) [B, M, C]
            outputs['pred_reg']: (Tensor) [B, M, 4]
            outputs['pred_ctn']: (Tensor) [B, M, 1]
            outputs['strides']: (List) [8, 16, 32, ...] stride of the model output
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
        """
        device = outputs['pred_cls'][0].device
        fpn_strides = outputs['strides']
        anchors = outputs['anchors']

        # matcher
        (
            gt_classes,
            gt_shifts_deltas,
            gt_centerness
        ) = self.matcher(fpn_strides, anchors, targets)

        # List[B, M, C] -> [B, M, C] -> [BM, C]
        pred_cls = torch.cat(outputs['pred_cls'], dim=1).view(-1, self.num_classes)
        pred_delta = torch.cat(outputs['pred_reg'], dim=1).view(-1, 4)
        pred_ctn = torch.cat(outputs['pred_ctn'], dim=1).view(-1, 1)
        masks = torch.cat(outputs['mask'], dim=1).view(-1)

        gt_classes = gt_classes.flatten().to(device)
        gt_shifts_deltas = gt_shifts_deltas.view(-1, 4).to(device)
        gt_centerness = gt_centerness.view(-1, 1).to(device)

        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foreground)
        num_foreground = torch.clamp(num_foreground / get_world_size(), min=1).item()

        num_foreground_centerness = gt_centerness[foreground_idxs].sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foreground_centerness)
        num_targets = torch.clamp(num_foreground_centerness / get_world_size(), min=1).item()

        gt_classes_target = torch.zeros_like(pred_cls)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # cls loss
        valid_idxs = (gt_classes >= 0) & masks
        loss_labels = self.loss_labels(
            pred_cls[valid_idxs],
            gt_classes_target[valid_idxs],
            num_boxes=num_foreground)

        # box loss
        loss_bboxes = self.loss_bboxes(
            pred_delta[foreground_idxs],
            gt_shifts_deltas[foreground_idxs],
            bbox_quality=gt_centerness[foreground_idxs],
            num_boxes=num_targets)

        # centerness loss
        loss_centerness = F.binary_cross_entropy_with_logits(pred_ctn[foreground_idxs], 
                                                             gt_centerness[foreground_idxs], 
                                                             reduction='none')

        loss_centerness = loss_centerness.sum() / num_foreground

        # total loss
        total_loss = self.loss_cls_weight * loss_labels + \
                     self.loss_reg_weight * loss_bboxes + \
                     self.loss_ctn_weight * loss_centerness

        loss_dict = dict(
                loss_labels = loss_labels,
                loss_bboxes = loss_bboxes,
                loss_centerness = loss_centerness,
                total_loss = total_loss
        )

        return loss_dict
    

    # Compute loss of FCOS with OTA matcher
    def ota_losses(self, outputs, targets):
        """
            outputs['pred_cls']: (Tensor) [B, M, C]
            outputs['pred_reg']: (Tensor) [B, M, 4]
            outputs['pred_ctn']: (Tensor) [B, M, 1]
            outputs['strides']: (List) [8, 16, 32, ...] stride of the model output
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
        """
        device = outputs['pred_cls'][0].device
        fpn_strides = outputs['strides']
        anchors = outputs['anchors']

        # matcher
        (
            gt_classes,
            gt_shifts_deltas,
            gt_ious
        ) = self.matcher(
            fpn_strides = fpn_strides, 
            anchors = anchors, 
            pred_cls_logits = outputs['pred_cls'], 
            pred_deltas = outputs['pred_reg'], 
            targets = targets
            )

        # List[B, M, C] -> [B, M, C] -> [BM, C]
        pred_cls = torch.cat(outputs['pred_cls'], dim=1).view(-1, self.num_classes)
        pred_delta = torch.cat(outputs['pred_reg'], dim=1).view(-1, 4)
        pred_iou = torch.cat(outputs['pred_ctn'], dim=1).view(-1, 1)
        masks = torch.cat(outputs['mask'], dim=1).view(-1)

        gt_classes = gt_classes.flatten().to(device)
        gt_shifts_deltas = gt_shifts_deltas.view(-1, 4).to(device)
        gt_ious = gt_ious.view(-1, 1).to(device)

        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foreground)
        num_foreground = torch.clamp(num_foreground / get_world_size(), min=1).item()

        gt_classes_target = torch.zeros_like(pred_cls)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # cls loss
        valid_idxs = (gt_classes >= 0) & masks
        loss_labels = self.loss_labels(
            pred_cls[valid_idxs],
            gt_classes_target[valid_idxs],
            num_boxes=num_foreground)

        # box loss
        loss_bboxes = self.loss_bboxes(
            pred_delta[foreground_idxs],
            gt_shifts_deltas[foreground_idxs],
            num_boxes=num_foreground)

        # iou loss
        loss_ious = F.binary_cross_entropy_with_logits(pred_iou[foreground_idxs], 
                                                       gt_ious[foreground_idxs], 
                                                       reduction='none')
        loss_ious = loss_ious.sum() / num_foreground

        # total loss
        total_loss = self.loss_cls_weight * loss_labels + \
                     self.loss_reg_weight * loss_bboxes + \
                     self.loss_ctn_weight * loss_ious

        loss_dict = dict(
                loss_labels = loss_labels,
                loss_bboxes = loss_bboxes,
                loss_ious = loss_ious,
                total_loss = total_loss
        )

        return loss_dict


    def __call__(self, outputs, targets):
        if self.cfg['matcher'] == 'matcher':
            return self.basic_losses(outputs, targets)
        elif self.cfg['matcher'] == 'ota_matcher':
            return self.ota_losses(outputs, targets)

# build criterion
def build_criterion(cfg, device, num_classes=80):
    criterion = Criterion(cfg=cfg, device=device, num_classes=num_classes)
    return criterion


if __name__ == "__main__":
    pass
