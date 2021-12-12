import numpy as np
import torch


def compute_iou(anchor_boxes, target_box):
    """
    Input:
        anchor_boxes : [N, 4]
        gt_box : [1, 4]
    Output:
        iou : [N, 4], where N = HW x KA
    """
    # anchor box: [HW x KA, 4]
    # convert  [xc, yc, w, h] -> [x1, y1, x2, y2]
    anchor_boxes_ = anchor_boxes.copy()
    anchor_boxes_[..., :2] = anchor_boxes[..., :2] - anchor_boxes[..., 2:] * 0.5 # x1y1
    anchor_boxes_[..., 2:] = anchor_boxes[..., :2] + anchor_boxes[..., 2:] * 0.5 # x2y2
    anchor_width = anchor_boxes_[:, 2] - anchor_boxes_[:, 0]
    anchor_height = anchor_boxes_[:, 3] - anchor_boxes_[:, 1]
    anchor_area = anchor_height * anchor_width
    
    # gt_box: [1, 4] -> [N, 4]
    target_box = np.repeat(target_box, anchor_boxes_.shape[0], axis=0)
    target_width = target_box[:, 2] - target_box[:, 0]
    target_height = target_box[:, 3] - target_box[:, 1]
    target_area = target_height * target_width

    # Area of intersection
    intersecion_width = np.minimum(anchor_boxes_[:, 2], target_box[:, 2]) - \
                            np.maximum(anchor_boxes_[:, 0], target_box[:, 0])
    intersection_height = np.minimum(anchor_boxes_[:, 3], target_box[:, 3]) - \
                            np.maximum(anchor_boxes_[:, 1], target_box[:, 1])
    intersection_area = intersection_height.clip(0.) * intersecion_width.clip(0.)
    # Area of union
    union_area = anchor_area + target_area - intersection_area + 1e-20
    # IoU
    iou = intersection_area / union_area

    return iou # [HW x KA]


def label_creator(targets,
                  anchor_boxes, 
                  num_classes, 
                  stride=32,
                  topk=8,
                  igt=0.15):
    """
        targets: (list of tensors) annotations
        anchor_boxes: (tensor) [1, HW, KA, 4]
        num_classes: (int) the number of class
        stride: (int) output stride of network
    """
    # prepare
    batch_size = len(targets)
    N, KA = anchor_boxes.shape[1:3]
    anchor_boxes = anchor_boxes

    # [B, HW x KA, cls+box+pos]
    target_tensor = np.zeros([batch_size, N*KA, num_classes + 4 + 1])
    # [1, HW, KA, 4] -> [HW x KA, 4]
    anchor_boxes = anchor_boxes.cpu().numpy().copy()[0].reshape(-1, 4)

    # generate gt datas  
    for bi in range(batch_size):
        target_i = targets[bi]
        boxes_i = target_i["boxes"].numpy()
        labels_i = target_i["labels"].numpy()

        for box, label in zip(boxes_i, labels_i):
            cls_id = int(label)
            x1, y1, x2, y2 = box
            gt_box = np.array([[x1, y1, x2, y2]])

            # compute IoU
            iou = compute_iou(anchor_boxes, gt_box)

            # keep the topk anchor boxes
            iou_sorted = -np.sort(-iou)
            iou_sorted_idx = np.argsort(-iou)

            # make labels
            for k in range(topk):
                iou_score = iou_sorted[k]
                if iou_score > igt:
                    grid_idx = iou_sorted_idx[k]
                    target_tensor[bi, grid_idx, :num_classes] = 0.0 # avoiding the multi labels for one grid cell
                    target_tensor[bi, grid_idx, cls_id] = 1.0
                    target_tensor[bi, grid_idx, num_classes:num_classes+4] = np.array([x1, y1, x2, y2])
                    target_tensor[bi, grid_idx, -1] = 1.0

    # [B, HW, KA, cls+box+pos]
    target_tensor = target_tensor.reshape(batch_size, N, KA,  num_classes + 4 + 1)
    
    return torch.from_numpy(target_tensor).float()


if __name__ == "__main__":
    gt_box = np.array([[0.0, 0.0, 10, 10]])
    anchor_boxes = np.array([[0.0, 0.0, 10, 10], 
                             [0.0, 0.0, 4, 4], 
                             [0.0, 0.0, 8, 8], 
                             [0.0, 0.0, 16, 16]
                             ])
    iou = compute_iou(anchor_boxes, gt_box)
    print(iou)
