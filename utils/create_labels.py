import numpy as np
import torch


def compute_iou(anchor_boxes, target_box):
    """
    Input:
        anchor_boxes : [HW, KA, 4]
        gt_box : [1, 4]
    Output:
        iou : [N, 4], where N = HW x KA
    """
    # anchor box: [HW, KA, 4] -> [HW x KA, 4]
    HW, KA = anchor_boxes.shape[:2]
    anchor_boxes = anchor_boxes.reshape(-1, 4)
    anchor_width = anchor_boxes[:, 2] - anchor_boxes[:, 0]
    anchor_height = anchor_boxes[:, 3] - anchor_boxes[:, 1]
    anchor_area = anchor_height * anchor_width
    
    # gt_box: [1, 4] -> [N, 4]
    target_box = np.repeat(target_box, anchor_boxes.shape[0], axis=0)
    target_width = target_box[:, 2] - target_box[:, 0]
    target_height = target_box[:, 3] - target_box[:, 1]
    target_area = target_height * target_width

    # Area of intersection
    intersecion_width = np.minimum(anchor_boxes[:, 2], target_box[:, 2]) - \
                            np.maximum(anchor_boxes[:, 0], target_box[:, 0])
    intersection_height = np.minimum(anchor_boxes[:, 3], target_box[:, 3]) - \
                            np.maximum(anchor_boxes[:, 1], target_box[:, 1])
    intersection_area = intersection_height * intersecion_width
    # Area of union
    union_area = anchor_area + target_area - intersection_area + 1e-20
    # IoU
    iou = intersection_area / union_area

    return iou # [HW x KA]


def label_creator(targets, 
                  anchor_boxes, 
                  num_classes, 
                  topk=8,
                  igt=0.15):
    """
        img_size: (int) the max line of the batch of images.size.
        stride: (int) the output stride of detector.
        targets: (list of tensors) annotations
        anchor_boxes: (tensor) [1, HW, KA, 4]
        num_classes: (int) the number of class 
    """
    # prepare
    batch_size = len(targets)
    num_queries = anchor_boxes.shape[1]
    anchor_boxes = anchor_boxes.cpu().numpy().copy()
    KA = anchor_boxes.shape[-2]

    # [B, HW x KA, cls+box+pos]
    target_tensor = np.zeros([batch_size, num_queries*KA, num_classes + 4 + 1])
    # [1, HW, KA, 4] -> [1, HW*KA, 4]
    anchor_boxes = anchor_boxes.reshape(anchor_boxes.shape[0], -1, 4)

    # generate gt datas  
    for bi in range(batch_size):
        target_i = targets[bi]
        boxes_i = target_i["boxes"]
        labels_i = target_i["labels"]

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
                    target_tensor[bi, grid_idx, :] = 0.0
                    target_tensor[bi, grid_idx, cls_id] = 1.0
                    target_tensor[bi, grid_idx, num_classes:num_classes+4] = np.array([x1, y1, x2, y2])
                    target_tensor[bi, grid_idx, -1] = 1.0

    # [B, HW, KA, cls+box+pos]
    target_tensor = target_tensor.reshape(batch_size, num_queries, KA,  num_classes + 4 + 1)
    
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
