import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import build_backbone
from .fpn import build_fpn
from .head import build_head

from utils.nms import multiclass_nms

DEFAULT_SCALE_CLAMP = np.log(1000.0 / 16)


class RetinaNet(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 trainable = False, 
                 topk = 1000):
        super(RetinaNet, self).__init__()
        self.cfg = cfg
        self.device = device
        self.stride = cfg['stride']
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        self.anchor_size = self.generate_anchor_sizes(cfg)  # [S, KA, 2]
        self.num_anchors = self.anchor_size.shape[1]

        #-------------------------- Network -----------------------------#
        ## backbone
        self.backbone, bk_dims = build_backbone(cfg=cfg, pretrained=trainable)

        ## fpn neck
        self.fpn = build_fpn(cfg, bk_dims, cfg['head_dim'])
                                     
        ## head
        self.head = build_head(cfg, num_classes, self.num_anchors)


    def generate_anchor_sizes(self, cfg):
        basic_anchor_size = cfg['anchor_config']['basic_size']
        anchor_aspect_ratio = cfg['anchor_config']['aspect_ratio']
        anchor_area_scale = cfg['anchor_config']['area_scale']

        num_scales = len(basic_anchor_size)
        num_anchors = len(anchor_aspect_ratio) * len(anchor_area_scale)
        anchor_sizes = []
        for size in basic_anchor_size:
            for ar in anchor_aspect_ratio:
                for s in anchor_area_scale:
                    ah, aw = size
                    area = ah * aw * s
                    anchor_sizes.append([np.sqrt(ar * area), np.sqrt(area / ar)])
        # [S * KA, 2] -> [S, KA, 2]
        anchor_sizes = torch.as_tensor(anchor_sizes).view(num_scales, num_anchors, 2)

        return anchor_sizes


    def generate_anchors(self, level, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        # [KA, 2]
        anchor_size = self.anchor_size[level]

        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
        # [HW, 2] -> [HW, 1, 2] -> [HW, KA, 2] 
        anchor_xy = anchor_xy[:, None, :].repeat(1, self.num_anchors, 1)
        anchor_xy *= self.stride[level]

        # [KA, 2] -> [1, KA, 2] -> [HW, KA, 2]
        anchor_wh = anchor_size[None, :, :].repeat(fmp_h*fmp_w, 1, 1)

        # [HW, KA, 4] -> [M, 4], M = HW x KA
        anchor_boxes = torch.cat([anchor_xy, anchor_wh], dim=-1)
        anchor_boxes = anchor_boxes.view(-1, 4).to(self.device)

        return anchor_boxes
        

    def decode_boxes(self, anchor_boxes, pred_reg):
        """
            anchor_boxes: (List[Tensor]) [1, M, 4] or [M, 4]
            pred_reg:     (List[Tensor]) [B, M, 4] or [M, 4]
        """
        # x = x_anchor + dx * w_anchor
        # y = y_anchor + dy * h_anchor
        pred_ctr_offset = pred_reg[..., :2] * anchor_boxes[..., 2:]
        if self.cfg['ctr_clamp'] is not None:
            pred_ctr_offset = torch.clamp(pred_ctr_offset,
                                        max=self.cfg['ctr_clamp'],
                                        min=-self.cfg['ctr_clamp'])
        pred_ctr_xy = anchor_boxes[..., :2] + pred_ctr_offset

        # w = w_anchor * exp(tw)
        # h = h_anchor * exp(th)
        pred_dwdh = pred_reg[..., 2:]
        pred_dwdh = torch.clamp(pred_dwdh, 
                                max=DEFAULT_SCALE_CLAMP)
        pred_wh = anchor_boxes[..., 2:] * pred_dwdh.exp()

        # convert [x, y, w, h] -> [x1, y1, x2, y2]
        pred_x1y1 = pred_ctr_xy - 0.5 * pred_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_wh
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box


    def post_process(self, cls_pred, reg_pred, anchors):
        """
        Input:
            cls_pred: List(Tensor) [[H x W, C], ...]
            reg_pred: List(Tensor) [[H x W, 4], ...]
            anchors:  List(Tensor) [[H x W, 2], ...]
        """
        all_scores = []
        all_labels = []
        all_bboxes = []
        
        for cls_pred_i, reg_pred_i, anchors_i in zip(cls_pred, reg_pred, anchors):
            # (H x W x C,)
            cls_pred_i = torch.sqrt(cls_pred_i.sigmoid()).flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk, reg_pred_i.size(0))

            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = cls_pred_i.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            scores = topk_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
            labels = topk_idxs % self.num_classes

            reg_pred_i = reg_pred_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]

            # decode bbox
            bboxes = self.decode_boxes(anchors_i, reg_pred_i)

            all_scores.append(scores)
            all_labels.append(labels)
            all_bboxes.append(bboxes)

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        bboxes = torch.cat(all_bboxes)

        # to cpu
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, False)

        return bboxes, scores, labels


    @torch.no_grad()
    def inference_single_image(self, x):
        img_h, img_w = x.shape[2:]
        # backbone
        feats = self.backbone(x)

        # fpn neck
        pyramid_feats = [feats[-3], feats[-2], feats[-1]]
        pyramid_feats = self.fpn(pyramid_feats)

        # shared head
        all_cls_preds = []
        all_reg_preds = []
        all_anchors = []
        for level, feat in enumerate(pyramid_feats):
            cls_pred, reg_pred = self.head(feat)

            _, _, H, W = cls_pred.size()
            fmp_size = [H, W]
            # [1, C, H, W] -> [C, H, W]
            cls_pred = cls_pred[0]
            reg_pred = reg_pred[0]

            # decode box
            # [1, C, H, W] -> [H, W, C] -> [M, C]
            cls_pred = cls_pred.permute(1, 2, 0).contiguous().view(-1, self.num_classes)
            reg_pred = reg_pred.permute(1, 2, 0).contiguous().view(-1, 4)

            # [M, 4]
            anchors = self.generate_anchors(level, fmp_size)

            all_cls_preds.append(cls_pred)
            all_reg_preds.append(reg_pred)
            all_anchors.append(anchors)

        # post process
        bboxes, scores, labels = self.post_process(
            all_cls_preds, all_reg_preds, all_anchors)

        # normalize bbox
        bboxes[..., [0, 2]] /= img_w
        bboxes[..., [1, 3]] /= img_h
        bboxes = bboxes.clip(0., 1.)

        return bboxes, scores, labels


    def forward(self, x, mask=None):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            # backbone
            feats = self.backbone(x)

            # fpn neck
            pyramid_feats = [feats[-3], feats[-2], feats[-1]]
            pyramid_feats = self.fpn(pyramid_feats)

            # shared head
            all_anchor_boxes = []
            all_cls_preds = []
            all_reg_preds = []
            all_masks = []
            for level, feat in enumerate(pyramid_feats):
                cls_pred, reg_pred = self.head(feat)

                B, _, H, W = cls_pred.size()
                fmp_size = [H, W]
                # [B, AC, H, W] -> [B, H, W, AC] -> [B, M, C]
                cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
                reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)

                # generate anchors: [M, 2]
                anchor_boxes = self.generate_anchors(level, fmp_size)
                all_anchor_boxes.append(anchor_boxes)

                all_cls_preds.append(cls_pred)
                all_reg_preds.append(reg_pred)

                if mask is not None:
                    # [B, H, W]
                    mask_i = torch.nn.functional.interpolate(mask[None], size=[H, W]).bool()[0]
                    # [B, H, W] -> [B, M]
                    mask_i = mask_i.flatten(1)
                    # [B, HW] -> [B, HW, KA] -> [B, M], M= HW x KA
                    mask_i = mask_i[..., None].repeat(1, 1, self.num_anchors).flatten(1)
                    
                    all_masks.append(mask_i)

            all_cls_preds = torch.cat(all_cls_preds, dim=1)
            all_reg_preds = torch.cat(all_reg_preds, dim=1)
            all_masks = torch.cat(all_masks, dim=1)

            # decode box: [M, 4]
            all_anchor_boxes = torch.cat(all_anchor_boxes)
            all_box_preds = self.decode_boxes(all_anchor_boxes[None], all_reg_preds)

            # output dict
            outputs = {"pred_cls": all_cls_preds,            # List [B, M, C]
                       "pred_box": all_box_preds,            # List [B, M, 4]
                       "anchor_boxes": all_anchor_boxes,     # List [B, M, 2]
                       "strides": self.stride,
                       "mask": all_masks}                    # List [B, M,]

            return outputs 
    