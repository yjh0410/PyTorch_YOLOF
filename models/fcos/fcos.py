import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import build_backbone
from .fpn import build_fpn
from .head import build_head


class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """
    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale


class FCOS(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 trainable = False, 
                 topk = 1000):
        super(FCOS, self).__init__()
        self.cfg = cfg
        self.device = device
        self.stride = cfg['stride']
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk

        #-------------------------- Network -----------------------------#
        ## backbone
        self.backbone, bk_dims = build_backbone(cfg=cfg, pretrained=trainable)

        ## fpn neck
        self.fpn = build_fpn(cfg, bk_dims, cfg['head_dim'])
                                     
        ## head
        self.head = build_head(cfg, num_classes)

        ## scale
        self.scales = nn.ModuleList([Scale() for _ in range(len(self.stride))])


    def generate_anchors(self, level, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
        anchor_xy *= self.stride[level]
        anchors = anchor_xy.to(self.device)

        return anchors
        

    def decode_boxes(self, anchors, pred_deltas):
        """
            anchors:  (List[Tensor]) [1, M, 2] or [M, 2]
            pred_reg: (List[Tensor]) [B, M, 4] or [M, 4] (l, t, r, b)
        """
        # x1 = x_anchor - l, x2 = x_anchor + r
        # y1 = y_anchor - t, y2 = y_anchor + b
        pred_x1y1 = anchors - pred_deltas[..., :2]
        pred_x2y2 = anchors + pred_deltas[..., 2:]
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box


    def nms(self, dets, scores):
        """"Pure Python NMS."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # compute iou
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def post_process(self, cls_pred, reg_pred, ctn_pred, anchors):
        """
        Input:
            cls_pred: List(Tensor) [[H x W, C], ...]
            reg_pred: List(Tensor) [[H x W, 4], ...]
            ctn_pred: List(Tensor) [[H x W, 1], ...]
            anchors:  List(Tensor) [[H x W, 2], ...]
        """
        all_scores = []
        all_labels = []
        all_bboxes = []
        
        for cls_pred_i, reg_pred_i, ctn_pred_i, anchors_i in zip(cls_pred, reg_pred, ctn_pred, anchors):
            # (H x W x C,)
            cls_pred_i = (torch.sqrt(cls_pred_i.sigmoid() * ctn_pred_i.sigmoid())).flatten()

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
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

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
        all_ctn_preds = []
        all_anchors = []
        for level, feat in enumerate(pyramid_feats):
            cls_pred, reg_pred, ctn_pred = self.head(feat)

            _, _, H, W = cls_pred.size()
            fmp_size = [H, W]
            # [1, C, H, W] -> [C, H, W]
            cls_pred = cls_pred[0]
            reg_pred = reg_pred[0]
            ctn_pred = ctn_pred[0]

            # decode box
            # [1, C, H, W] -> [H, W, C] -> [M, C]
            cls_pred = cls_pred.permute(1, 2, 0).contiguous().view(-1, self.num_classes)
            reg_pred = reg_pred.permute(1, 2, 0).contiguous().view(-1, 4)
            reg_pred = F.relu(self.scales[level](reg_pred)) * self.stride[level]
            ctn_pred = ctn_pred.permute(1, 2, 0).contiguous().view(-1, 1)

            # [M, 4]
            anchors = self.generate_anchors(level, fmp_size)

            all_cls_preds.append(cls_pred)
            all_reg_preds.append(reg_pred)
            all_ctn_preds.append(ctn_pred)
            all_anchors.append(anchors)

        # post process
        bboxes, scores, labels = self.post_process(all_cls_preds, all_reg_preds, all_ctn_preds, all_anchors)

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
            all_anchors = []
            all_cls_preds = []
            all_reg_preds = []
            all_ctn_preds = []
            all_masks = []
            for level, feat in enumerate(pyramid_feats):
                cls_pred, reg_pred, ctn_pred = self.head(feat)

                B, _, H, W = cls_pred.size()
                fmp_size = [H, W]
                # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
                cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
                reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
                reg_pred = F.relu(self.scales[level](reg_pred)) * self.stride[level]
                ctn_pred = ctn_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)

                # generate anchors: [M, 2]
                anchors = self.generate_anchors(level, fmp_size)
                all_anchors.append(anchors)
            
                if self.cfg['decode_bbox']:
                    # decode bbox
                    reg_pred = self.decode_boxes(anchors, reg_pred)

                all_cls_preds.append(cls_pred)
                all_reg_preds.append(reg_pred)
                all_ctn_preds.append(ctn_pred)

                if mask is not None:
                    # [B, H, W]
                    mask_i = torch.nn.functional.interpolate(mask[None], size=[H, W]).bool()[0]
                    # [B, H, W] -> [B, M]
                    mask_i = mask_i.flatten(1)
                    
                    all_masks.append(mask_i)

            # output dict
            outputs = {"pred_cls": all_cls_preds,  # List [B, M, C]
                       "pred_reg": all_reg_preds,  # List [B, M, 4]
                       "pred_ctn": all_ctn_preds,  # List [B, M, 1]
                       "anchors": all_anchors,     # List [B, M, 2]
                       "strides": self.stride,
                       "mask": all_masks}          # List [B, M,]

            return outputs 
    