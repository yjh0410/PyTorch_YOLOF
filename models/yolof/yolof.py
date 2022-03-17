import numpy as np
import math
import torch
import torch.nn as nn
from ..backbone import build_backbone
from ..neck import build_neck
from ..head.decoupled_head import DecoupledHead


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

        # backbone
        self.backbone, bk_dim = build_backbone(model_name=cfg['backbone'], 
                                               pretrained=trainable,
                                               norm_type=cfg['norm_type'])

        # neck
        self.neck = build_neck(cfg=cfg, 
                               in_dim=bk_dim, 
                               out_dim=cfg['head_dim'])
                                     
        # head
        self.head = DecoupledHead(head=cfg['head'],
                                  head_dim=cfg['head_dim'],
                                  kernel_size=3,
                                  padding=1,
                                  num_classes=num_classes,
                                  trainable=trainable,
                                  num_anchors=self.num_anchors,
                                  act_type=cfg['act_type'])


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
            anchor_xy = anchor_xy[:, None, :].repeat(1, self.num_anchors, 1).to(self.device)
            anchor_xy *= self.stride

            # [KA, 2] -> [1, KA, 2] -> [HW, KA, 2]
            anchor_wh = self.anchor_size[None, :, :].repeat(fmp_h*fmp_w, 1, 1).to(self.device)

            # [HW, KA, 4] -> [M, 4]
            anchor_boxes = torch.cat([anchor_xy, anchor_wh], dim=-1)
            anchor_boxes = anchor_boxes.view(-1, 4)

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


    def postprocess(self, bboxes, scores):
        """
        bboxes: (N, 4), bsize = 1
        scores: (N, C), bsize = 1
        """

        cls_inds = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bboxes, scores, cls_inds


    @torch.no_grad()
    def inference_single_image(self, x):
        img_h, img_w = x.shape[2:]
        # backbone
        x = self.backbone(x)

        # neck
        x = self.neck(x)
        H, W = x.shape[2:]

        # head
        cls_pred, reg_pred = self.head(x)

        # decode box
        anchor_boxes = self.generate_anchors(fmp_size=[H, W]) # [M, 4]

        # scores
        scores, labels = torch.max(cls_pred.sigmoid(), dim=-1)

        # topk
        if scores.shape[0] > self.topk:
            scores, indices = torch.topk(scores, self.topk)
            labels = labels[indices]
            reg_pred = reg_pred[indices]
            anchor_boxes = anchor_boxes[indices]

        # decode box
        bboxes = self.decode_boxes(anchor_boxes[None], reg_pred[None])[0] # [N, 4]

        # to cpu
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # threshold
        keep = np.where(scores >= self.conf_thresh)
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]

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
            x = self.backbone(x)

            # neck
            x = self.neck(x)
            H, W = x.shape[2:]

            # head
            cls_pred, reg_pred = self.head(x)

            # decode box
            anchor_boxes = self.generate_anchors(fmp_size=[H, W]) # [M, 4]
            box_pred = self.decode_boxes(anchor_boxes[None], reg_pred) # [B, M, 4]
            
            if mask is not None:
                # [B, H, W]
                mask = torch.nn.functional.interpolate(mask[None], size=[H, W]).bool()[0]
                # [B, H, W, KA] -> [B, HW]
                mask = mask.flatten(1)
                # [B, HW, KA] -> [BM,], M= HW x KA
                mask = mask[..., None].repeat(1, 1, self.num_anchors).flatten()

            outputs = {"pred_cls": cls_pred,
                       "pred_box": box_pred,
                       "mask": mask}

            return outputs 
