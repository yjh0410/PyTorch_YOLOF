import torch
import torch.nn as nn
import numpy as np
import os
import cv2

from data.voc import VOCDetection
from data.coco import COCODataset
from data.transforms import TrainTransforms, ValTransforms, BaseTransforms
from evaluator.coco_evaluator import COCOAPIEvaluator
from evaluator.voc_evaluator import VOCAPIEvaluator


def vis_data(images, targets, masks):
    """
        images: (tensor) [B, 3, H, W]
        targets: (list) a list of targets
        masks: (tensor) [B, H, W]
    """
    batch_size = images.size(0)
    # vis data
    rgb_mean = [123.675, 116.28, 103.53]
    rgb_std = [58.395, 57.12, 57.375]

    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(91)]

    for bi in range(batch_size):
        # mask
        mask = masks[bi].bool()
        image_tensor = images[bi]
        index = torch.nonzero(mask)

        # pad image
        # to numpy
        pad_image = image_tensor.permute(1, 2, 0).cpu().numpy()
        # denormalize
        pad_image = (pad_image * rgb_std + rgb_mean).astype(np.uint8)
        # to BGR
        pad_image = pad_image[..., (2, 1, 0)]

        # valid image without pad
        valid_image = image_tensor[:, :index[-1, 0]+1, :index[-1, 1]+1]
        valid_image = valid_image.permute(1, 2, 0).cpu().numpy()
        valid_image = (valid_image * rgb_std + rgb_mean).astype(np.uint8)
        valid_image = valid_image[..., (2, 1, 0)]

        valid_image = valid_image.copy()

        targets_i = targets[bi]
        tgt_boxes = targets_i['boxes']
        tgt_labels = targets_i['labels']

        # to numpy
        mask = mask.cpu().numpy() * 255
        mask = mask.astype(np.uint8)

        
        for box, label in zip(tgt_boxes, tgt_labels):
            x1, y1, x2, y2 = box
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cls_id = int(label)
            color = class_colors[cls_id]

            valid_image = cv2.rectangle(valid_image, (x1, y1), (x2, y2), color, 2)

        cv2.imshow('pad image', pad_image)
        cv2.waitKey(0)

        cv2.imshow('valid image', valid_image)
        cv2.waitKey(0)

        cv2.imshow('mask', mask)
        cv2.waitKey(0)


def build_dataset(cfg, args, device):
    # transform
    trans_config = cfg['transforms'][args.schedule]
    print('==============================')
    print('TrainTransforms: {}'.format(trans_config))
    train_transform = TrainTransforms(trans_config=trans_config,
                                      min_size=cfg['train_min_size'],
                                      max_size=cfg['train_max_size'],
                                      random_size=cfg['epoch'][args.schedule]['multi_scale'],
                                      pixel_mean=cfg['pixel_mean'],
                                      pixel_std=cfg['pixel_std'],
                                      format=cfg['format'])
    val_transform = ValTransforms(min_size=cfg['test_min_size'],
                                  max_size=cfg['test_max_size'],
                                  pixel_mean=cfg['pixel_mean'],
                                  pixel_std=cfg['pixel_std'],
                                  format=cfg['format'])
    color_augment = BaseTransforms(min_size=cfg['train_min_size'],
                                   max_size=cfg['train_max_size'],
                                   random_size=cfg['epoch'][args.schedule]['multi_scale'],
                                   pixel_mean=cfg['pixel_mean'],
                                   pixel_std=cfg['pixel_std'],
                                   format=cfg['format'])
    # dataset
    
    if args.dataset == 'voc':
        data_dir = os.path.join(args.root, 'VOCdevkit')
        num_classes = 20
        # dataset
        dataset = VOCDetection(img_size=cfg['train_max_size'],
                               data_dir=data_dir, 
                               transform=train_transform,
                               color_augment=color_augment,
                               mosaic=cfg['mosaic'])
        # evaluator
        evaluator = VOCAPIEvaluator(data_dir=data_dir,
                                    device=device,
                                    transform=val_transform)

    elif args.dataset == 'coco':
        data_dir = os.path.join(args.root, 'COCO')
        num_classes = 80
        # dataset
        dataset = COCODataset(img_size=cfg['train_max_size'],
                              data_dir=data_dir,
                              image_set='train2017',
                              transform=train_transform,
                              color_augment=color_augment,
                              mosaic=cfg['mosaic'])
        # evaluator
        evaluator = COCOAPIEvaluator(data_dir=data_dir,
                                     device=device,
                                     transform=val_transform)
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    print('==============================')
    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))

    return dataset, evaluator, num_classes


def build_dataloader(args, dataset, collate_fn=None):
    # distributed
    if args.distributed:
        # dataloader
        dataloader = torch.utils.data.DataLoader(
                        dataset=dataset, 
                        batch_size=args.batch_size, 
                        collate_fn=collate_fn,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        sampler=torch.utils.data.distributed.DistributedSampler(dataset)
                        )

    else:
        # dataloader
        dataloader = torch.utils.data.DataLoader(
                        dataset=dataset, 
                        shuffle=True,
                        batch_size=args.batch_size, 
                        collate_fn=collate_fn,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        drop_last=True
                        )
    return dataloader
    

def nms(dets, scores, nms_thresh=0.4):
    """"Pure Python NMS baseline."""
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
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-28, xx2 - xx1)
        h = np.maximum(1e-28, yy2 - yy1)
        inter = w * h

        # Cross Area / (bbox + particular area - Cross Area)
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
        #reserve all the boundingbox whose ovr less than thresh
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    return keep


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm


def load_weight(model, path_to_ckpt=None):
    if path_to_ckpt is None:
        print('No weight file ...')
        return model
    else:
        checkpoint = torch.load(path_to_ckpt, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict)

        return model


class CollateFunc(object):
    def _max_by_axis(self, the_list):
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes


    def __call__(self, batch):
        batch = list(zip(*batch))

        image_list = batch[0]
        target_list = batch[1]

        # TODO make this more general
        if image_list[0].ndim == 3:

            # TODO make it support different-sized images
            max_size = self._max_by_axis([list(img.shape) for img in image_list])
            # min_size = tuple(min(s) for s in zip(*[img.shape for img in image_list]))
            batch_shape = [len(image_list)] + max_size
            b, c, h, w = batch_shape
            dtype = image_list[0].dtype
            device = image_list[0].device
            batch_tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            batch_mask = torch.zeros((b, h, w), dtype=dtype, device=device)

            for img, pad_img, m in zip(image_list, batch_tensor, batch_mask):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                m[: img.shape[1], :img.shape[2]] = 1.0
        else:
            raise ValueError('not supported')
            
        return batch_tensor, target_list, batch_mask


# test time augmentation(TTA)
class TestTimeAugmentation(object):
    def __init__(self, num_classes=80, nms_thresh=0.4, scale_range=[320, 640, 32]):
        self.nms = nms
        self.num_classes = num_classes
        self.nms_thresh = nms_thresh
        self.scales = np.arange(scale_range[0], scale_range[1]+1, scale_range[2])
        
    def __call__(self, x, model):
        # x: Tensor -> [B, C, H, W]
        bboxes_list = []
        scores_list = []
        labels_list = []

        # multi scale
        for s in self.scales:
            if x.size(-1) == s and x.size(-2) == s:
                x_scale = x
            else:
                x_scale =torch.nn.functional.interpolate(
                                        input=x, 
                                        size=(s, s), 
                                        mode='bilinear', 
                                        align_corners=False)
            model.set_grid(s)
            bboxes, scores, labels = model(x_scale)
            bboxes_list.append(bboxes)
            scores_list.append(scores)
            labels_list.append(labels)

            # Flip
            x_flip = torch.flip(x_scale, [-1])
            bboxes, scores, labels = model(x_flip)
            bboxes = bboxes.copy()
            bboxes[:, 0::2] = 1.0 - bboxes[:, 2::-2]
            bboxes_list.append(bboxes)
            scores_list.append(scores)
            labels_list.append(labels)

        bboxes = np.concatenate(bboxes_list)
        scores = np.concatenate(scores_list)
        labels = np.concatenate(labels_list)

        # nms
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores, self.nms_thresh)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        return bboxes, scores, labels
