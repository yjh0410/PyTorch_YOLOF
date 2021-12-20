import numpy as np
import cv2


def vis_data(images, targets, masks):
    """
        images: (tensor) [B, 3, H, W]
        targets: (list) a list of targets
        masks: (tensor) [B, H, W]
    """
    batch_size = images.size(0)
    # vis data
    rgb_mean=np.array((0.485, 0.456, 0.406), dtype=np.float32)
    rgb_std=np.array((0.229, 0.224, 0.225), dtype=np.float32)

    for bi in range(batch_size):
        # to numpy
        image = images[bi].permute(1, 2, 0).cpu().numpy()
        # denormalize
        image = ((image * rgb_std + rgb_mean)*255).astype(np.uint8)
        # to BGR
        image = image[..., (2, 1, 0)]
        image = image.copy()

        boxes = targets[bi]["boxes"]
        labels = targets[bi]["labels"]
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        cv2.imshow('groundtruth', image)
        cv2.waitKey(0)

        # to numpy
        mask = masks[bi].cpu().numpy()
        mask = (mask * 255).astype(np.uint8).copy()

        boxes = targets[bi]["boxes"]
        labels = targets[bi]["labels"]
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        cv2.imshow('mask', mask)
        cv2.waitKey(0)


def vis_targets(images, targets, anchor_boxes, stride=32):
    """
        images: (tensor) [B, 3, H, W]
        targets: (tensor) [B, HW, KA, C+4+1]
        anchor_boxes: (tensor) [1, HW, KA, 4]
    """
    batch_size = images.size(0)
    # vis data
    rgb_mean=np.array((0.485, 0.456, 0.406), dtype=np.float32)
    rgb_std=np.array((0.229, 0.224, 0.225), dtype=np.float32)
    # [1, HW, KA, 4] -> [HW, KA, 4]
    anchor_boxes = anchor_boxes.cpu().numpy().copy()[0]

    for bi in range(batch_size):
        # to numpy
        image = images[bi].permute(1, 2, 0).cpu().numpy()
        # denormalize
        image = ((image * rgb_std + rgb_mean)*255).astype(np.uint8)
        # to BGR
        image = image[..., (2, 1, 0)]
        image1 = image.copy()
        image2 = image.copy()

        target_i = targets[bi] # [HW, KA, C+4+1]
        for j in range(target_i.shape[0]):
            for k in range(target_i.shape[1]):
                t = target_i[j, k] # [C+4+1]
                if t[-1].item() > 0.: # positive sample
                    # gt box
                    box = t[-5:-1]
                    x1s, y1s, x2s, y2s = box
                    x1 = int(x1s * stride)
                    y1 = int(y1s * stride)
                    x2 = int(x2s * stride)
                    y2 = int(y2s * stride)
                    cv2.rectangle(image1, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.rectangle(image2, (x1, y1), (x2, y2), (0, 0, 255), 3)

                    # positive anchor box
                    ab_box = anchor_boxes[j, k]
                    xcs, ycs, ws, hs = ab_box
                    x1 = int((xcs - ws / 2.0) * stride)
                    y1 = int((ycs - hs / 2.0) * stride)
                    x2 = int((xcs + ws / 2.0) * stride)
                    y2 = int((ycs + hs / 2.0) * stride)
                    cv2.rectangle(image1, (x1, y1), (x2, y2), (255, 0, 0), 3)
                elif t[-1].item() == -1.0: # ignore sample
                    # ignore anchor box
                    ab_box = anchor_boxes[j, k]
                    xcs, ycs, ws, hs = ab_box
                    x1 = int((xcs - ws / 2.0) * stride)
                    y1 = int((ycs - hs / 2.0) * stride)
                    x2 = int((xcs + ws / 2.0) * stride)
                    y2 = int((ycs + hs / 2.0) * stride)
                    cv2.rectangle(image2, (x1, y1), (x2, y2), (0, 255, 0), 3)

        image_ = np.concatenate([image1, image2], axis=1)
        cv2.namedWindow("assignment", 0)
        cv2.resizeWindow("assignment", image1.shape[1], image1.shape[0])
        cv2.imshow('assignment', image_)
        cv2.waitKey(0)


def vis_anchor_boxes(images, anchor_boxes, stride=32):
    """
        images: (tensor) [B, 3, H, W]
        anchor_boxes: (tensor) [1, HW, KA, 4]
    """
    batch_size = images.size(0)
    # vis data
    rgb_mean=np.array((0.485, 0.456, 0.406), dtype=np.float32)
    rgb_std=np.array((0.229, 0.224, 0.225), dtype=np.float32)
    # [1, HW, KA, 4] -> [HW x KA, 4]
    anchor_boxes = anchor_boxes.cpu().numpy().copy()[0].reshape(-1, 4)

    for bi in range(batch_size):
        # to numpy
        image = images[bi].permute(1, 2, 0).cpu().numpy()
        # denormalize
        image = ((image * rgb_std + rgb_mean)*255).astype(np.uint8)
        # to BGR
        image = image[..., (2, 1, 0)]
        image = image.copy()

        for ab_box in anchor_boxes:
            xcs, ycs, ws, hs = ab_box
            x1 = int((xcs - ws / 2.0) * stride)
            y1 = int((ycs - hs / 2.0) * stride)
            x2 = int((xcs + ws / 2.0) * stride)
            y2 = int((ycs + hs / 2.0) * stride)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

        cv2.imshow('anchor boxes', image)
        cv2.waitKey(0)
