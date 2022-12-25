import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import SimOTA
from utils.box_ops import get_ious
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
        # matcher
        matcher_config = cfg['matcher']
        self.matcher = SimOTA(
            num_classes=num_classes,
            center_sampling_radius=matcher_config['center_sampling_radius'],
            topk_candidate=matcher_config['topk_candicate']
            )


    def __call__(self, outputs, targets):        
        """
            outputs['pred_obj']: List(Tensor) [B, M, 1]
            outputs['pred_cls']: List(Tensor) [B, M, C]
            outputs['pred_box']: List(Tensor) [B, M, 4]
            outputs['strides']: List(Int) [8, 16, 32] output stride
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
        """
        bs = outputs['pred_cls'].shape[0]
        device = outputs['pred_cls'].device
        stride = outputs['stride']
        anchors = outputs['anchors']
        num_anchors = anchors.shape[0]
        # preds: [B, M, C]
        cls_pred = outputs['pred_cls']
        box_pred = outputs['pred_box']

        # label assignment
        cls_targets = []
        box_targets = []
        fg_masks = []

        for batch_idx in range(bs):
            tgt_labels = targets[batch_idx]["labels"].to(device)
            tgt_bboxes = targets[batch_idx]["boxes"].to(device)

            # check target
            if len(tgt_labels) == 0 or tgt_bboxes.max().item() == 0.:
                # There is no valid gt
                cls_target = cls_pred.new_zeros((0, self.num_classes))
                box_target = cls_pred.new_zeros((0, 4))
                fg_mask = cls_pred.new_zeros(num_anchors).bool()
            else:
                (
                    gt_matched_classes,
                    fg_mask,
                    matched_gt_inds,
                ) = self.matcher(
                    stride = stride,
                    anchors = anchors,
                    pred_cls = cls_pred[batch_idx], 
                    pred_box = box_pred[batch_idx],
                    tgt_labels = tgt_labels,
                    tgt_bboxes = tgt_bboxes
                    )
                # class target
                tgt_cls_ot = F.one_hot(gt_matched_classes.long(), self.num_classes)
                cls_target = cls_pred.new_zeros((num_anchors, self.num_classes))
                cls_target[fg_mask] = tgt_cls_ot.float()
                # box target
                box_target = tgt_bboxes[matched_gt_inds]

            cls_targets.append(cls_target)
            box_targets.append(box_target)
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        box_targets = torch.cat(box_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        num_foregrounds = fg_masks.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foregrounds)
        num_foregrounds = (num_foregrounds / get_world_size()).clamp(1.0)
        
        # classification loss
        valid_idxs = outputs['mask'].view(-1)
        print(valid_idxs.shape, outputs['mask'].shape)
        print(cls_pred.shape, cls_targets.shape)
        cls_pred = cls_pred.view(-1, self.num_classes)
        loss_labels = sigmoid_focal_loss(
            cls_pred[valid_idxs], cls_targets[valid_idxs],
            self.alpha, self.gamma, reduction='none')
        loss_labels = loss_labels.sum() / num_foregrounds

        # regression loss
        matched_box_pred = box_pred.view(-1, 4)[fg_masks]
        ious = get_ious(matched_box_pred,
                        box_targets,
                        box_mode="xyxy",
                        iou_type='giou')
        loss_bboxes = (1.0 - ious).sum() / num_foregrounds

        # total loss
        losses = self.loss_cls_weight * loss_labels + \
                 self.loss_reg_weight * loss_bboxes

        loss_dict = dict(
                loss_labels = loss_labels,
                loss_bboxes = loss_bboxes,
                total_loss = losses
        )

        return loss_dict
    

def build_criterion(cfg, device, num_classes):
    criterion = Criterion(
        cfg=cfg,
        device=device,
        num_classes=num_classes
        )

    return criterion


if __name__ == "__main__":
    pass