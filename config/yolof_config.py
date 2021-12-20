# YOLOF config


yolof_config = {
    # 1x
    'yolof_r50_C5_1x': {
        # model
        'backbone': 'resnet50',
        'dilated_block': [2, 4, 6, 8],
        'head_dims': 512,
        'bottle_ratio': 0.25,
        'anchor_size': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # epoch
        'max_epoch': 12,
        'lr_epoch': [8, 11],
        # matcher
        'topk': 4,
        'iou_thresh': 0.15,
        'ignore_thresh': 0.7
    },
    'yolof_r101_C5_1x': {
        # model
        'backbone': 'resnet101',
        'dilated_block': [2, 4, 6, 8],
        'head_dims': 512,
        'bottle_ratio': 0.25,
        'anchor_size': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # epoch
        'max_epoch': 12,
        'lr_epoch': [8, 11],
        # matcher
        'topk': 4,
        'iou_thresh': 0.15,
        'ignore_thresh': 0.7
    },
    'yolof_r50_DC5_1x': {
        # model
        'backbone': 'resnet50-d',
        'dilated_block': [4, 8, 12, 16],
        'head_dims': 512,
        'bottle_ratio': 0.25,
        'anchor_size': [[16, 16], [32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # epoch
        'max_epoch': 12,
        'lr_epoch': [8, 11],
        # matcher
        'topk': 4,
        'iou_thresh': 0.15,
        'ignore_thresh': 0.7
    },
    'yolof_r101_DC5_1x': {
        # model
        'backbone': 'resnet101-d',
        'dilated_block': [4, 8, 12, 16],
        'head_dims': 512,
        'bottle_ratio': 0.25,
        'anchor_size': [[16, 16], [32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # epoch
        'max_epoch': 12,
        'lr_epoch': [8, 11],
        # matcher
        'topk': 4,
        'iou_thresh': 0.15,
        'ignore_thresh': 0.7
    }
}