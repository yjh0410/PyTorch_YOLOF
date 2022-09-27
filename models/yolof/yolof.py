import numpy as np
import math
import torch
import torch.nn as nn
from ..backbone import build_backbone
from .encoder import build_encoder
from .decoder import build_decoder


DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


class YOLOF(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 trainable = False, 
                 topk = 1000):
        super(YOLOF, self).__init__()
        self.cfg = cfg
        self.device = device
        self.fmp_size = None
        self.stride = cfg['stride']
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        self.anchor_size = torch.as_tensor(cfg['anchor_size'])
        self.num_anchors = len(cfg['anchor_size'])

        #-------------------------- Network -----------------------------#
        ## backbone
        self.backbone, bk_dim = build_backbone(cfg=cfg, pretrained=trainable)

        ## neck
        self.neck = build_encoder(cfg=cfg, in_dim=bk_dim, out_dim=cfg['encoder_dim'])
                                     
        ## head
        self.head = build_decoder(cfg, cfg['encoder_dim'], num_classes, self.num_anchors)


    def generate_anchors(self, fmp_size):
        """fmp_size: list -> [H, W] \n
           stride: int -> output stride
        """
        # check anchor boxes
        if self.fmp_size is not None and self.fmp_size == fmp_size:
            return self.anchor_boxes
        else:
            # generate grid cells
            fmp_h, fmp_w = fmp_size
            anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
            # [H, W, 2] -> [HW, 2]
            anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
            # [HW, 2] -> [HW, 1, 2] -> [HW, KA, 2] 
            anchor_xy = anchor_xy[:, None, :].repeat(1, self.num_anchors, 1)
            anchor_xy *= self.stride

            # [KA, 2] -> [1, KA, 2] -> [HW, KA, 2]
            anchor_wh = self.anchor_size[None, :, :].repeat(fmp_h*fmp_w, 1, 1)

            # [HW, KA, 4] -> [M, 4]
            anchor_boxes = torch.cat([anchor_xy, anchor_wh], dim=-1)
            anchor_boxes = anchor_boxes.view(-1, 4).to(self.device)

            self.anchor_boxes = anchor_boxes
            self.fmp_size = fmp_size

            return anchor_boxes
        

    def decode_boxes(self, anchor_boxes, pred_reg):
        """
            anchor_boxes: (List[tensor]) [1, M, 4]
            pred_reg: (List[tensor]) [B, M, 4]
        """
        # x = x_anchor + dx * w_anchor
        # y = y_anchor + dy * h_anchor
        pred_ctr_offset = pred_reg[..., :2] * anchor_boxes[..., 2:]
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

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def post_process(self, cls_pred, reg_pred, anchors):
        """
        Input:
            cls_pred: (Tensor) [H x W x KA, C]
            reg_pred: (Tensor) [H x W x KA, 4]
            anchors:  (Tensor) [H x W x KA, 4]
        """
        # (HxWxAxK,)
        cls_pred = cls_pred.flatten().sigmoid_()

        # Keep top k top scoring indices only.
        num_topk = min(self.topk, reg_pred.size(0))
        # torch.sort is actually faster than .topk (at least on GPUs)
        predicted_prob, topk_idxs = cls_pred.sort(descending=True)
        predicted_prob = predicted_prob[:num_topk]
        topk_idxs = topk_idxs[:num_topk]

        # filter out the proposals with low confidence score
        keep_idxs = predicted_prob > self.conf_thresh
        predicted_prob = predicted_prob[keep_idxs]
        topk_idxs = topk_idxs[keep_idxs]

        anchor_idxs = topk_idxs // self.num_classes
        labels = topk_idxs % self.num_classes

        reg_pred = reg_pred[anchor_idxs]
        anchors = anchors[anchor_idxs]

        # decode bbox
        bboxes = self.decode_boxes(anchors, reg_pred)

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
        xs = self.backbone(x)
        x = xs[-1]

        # neck
        x = self.neck(x)
        fmp_h, fmp_w = x.shape[2:]

        # head
        cls_pred, reg_pred = self.head(x)
        cls_pred, reg_pred = cls_pred[0], reg_pred[0]

        # anchor box
        anchor_boxes = self.generate_anchors(fmp_size=[fmp_h, fmp_w]) # [M, 4]

        # post process
        bboxes, scores, labels = self.post_process(cls_pred, reg_pred, anchor_boxes)

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
            xs = self.backbone(x)
            x = xs[-1]

            # neck
            x = self.neck(x)
            fmp_h, fmp_w = x.shape[2:]

            # head
            cls_pred, reg_pred = self.head(x)

            # anchor box: [M, 4]
            anchor_boxes = self.generate_anchors(fmp_size=[fmp_h, fmp_w])

            # decode box: [B, M, 4]
            box_pred = self.decode_boxes(anchor_boxes[None], reg_pred)
            
            if mask is not None:
                # [B, H, W]
                mask = torch.nn.functional.interpolate(mask[None], size=[fmp_h, fmp_w]).bool()[0]
                # [B, H, W] -> [B, HW]
                mask = mask.flatten(1)
                # [B, HW] -> [B, HW, KA] -> [BM,], M= HW x KA
                mask = mask[..., None].repeat(1, 1, self.num_anchors).flatten()

            outputs = {"pred_cls": cls_pred,
                       "pred_box": box_pred,
                       "anchors": anchor_boxes,
                       "mask": mask}

            return outputs 
