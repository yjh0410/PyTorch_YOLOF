# YOLOF config


yolof_config = {
    'yolof18': {
        # input
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': {
            '1x':[{'name': 'ToTensor'},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'RandomShift', 'max_shift': 32},
                    {'name': 'Resize'},
                    {'name': 'Normalize'},
                    {'name': 'PadImage'}],

            '2x':[{'name': 'ToTensor'},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'RandomShift', 'max_shift': 32},
                    {'name': 'Resize'},
                    {'name': 'Normalize'},
                    {'name': 'PadImage'}],

            '3x':[{'name': 'DistortTransform'},
                    {'name': 'ToTensor'},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'RandomShift', 'max_shift': 32},
                    {'name': 'RandomSizeCrop'},                 
                    {'name': 'Resize'},
                    {'name': 'Normalize'},
                    {'name': 'PadImage'}]},
        # model
        'backbone': 'resnet18',
        'norm_type': 'FrozeBN',
        'stride': 32,
        'act_type': 'relu',
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [2, 4, 6, 8],
        'expand_ratio': 0.25,
        # head
        'head_dim': 512,
        'head': 'naive_head',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # anchor box
        'anchor_size': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # matcher
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1500,
        'warmup_factor': 0.00066667,
        'epoch': {
            '1x': {'max_epoch': 12, 
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24, 
                    'lr_epoch': [16, 22], 
                    'multi_scale': [480, 544, 608, 672, 736, 800]},
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [480, 544, 608, 672, 736, 800]},
        },
    },

    'yolof50': {
        # input
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': {
            '1x':[{'name': 'ToTensor'},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'RandomShift', 'max_shift': 32},
                    {'name': 'Resize'},
                    {'name': 'Normalize'},
                    {'name': 'PadImage'}],

            '2x':[{'name': 'ToTensor'},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'RandomShift', 'max_shift': 32},
                    {'name': 'Resize'},
                    {'name': 'Normalize'},
                    {'name': 'PadImage'}],

            '3x':[{'name': 'DistortTransform'},
                    {'name': 'ToTensor'},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'RandomShift', 'max_shift': 32},
                    {'name': 'RandomSizeCrop'},                 
                    {'name': 'Resize'},
                    {'name': 'Normalize'},
                    {'name': 'PadImage'}]},
        # model
        'backbone': 'resnet50',
        'norm_type': 'FrozeBN',
        'stride': 32,
        'act_type': 'relu',
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [2, 4, 6, 8],
        'expand_ratio': 0.25,
        # head
        'head_dim': 512,
        'head': 'naive_head',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # anchor box
        'anchor_size': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # matcher
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1500,
        'warmup_factor': 0.00066667,
        'epoch': {
            '1x': {'max_epoch': 12, 
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24, 
                    'lr_epoch': [16, 22], 
                    'multi_scale': [480, 544, 608, 672, 736, 800]},
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [480, 544, 608, 672, 736, 800]},
        },
    },

    'yolof50-DC5': {
        # input
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': {
            '1x':[{'name': 'ToTensor'},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'RandomShift', 'max_shift': 32},
                    {'name': 'Resize'},
                    {'name': 'Normalize'},
                    {'name': 'PadImage'}],

            '2x':[{'name': 'ToTensor'},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'RandomShift', 'max_shift': 32},
                    {'name': 'Resize'},
                    {'name': 'Normalize'},
                    {'name': 'PadImage'}],

            '3x':[{'name': 'DistortTransform'},
                    {'name': 'ToTensor'},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'RandomShift', 'max_shift': 32},
                    {'name': 'RandomSizeCrop'},                 
                    {'name': 'Resize'},
                    {'name': 'Normalize'},
                    {'name': 'PadImage'}]},
        # model
        'backbone': 'resnet50-d',
        'norm_type': 'FrozeBN',
        'stride': 16,
        'act_type': 'relu',
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [4, 8, 12, 16],
        'expand_ratio': 0.25,
        # head
        'head_dim': 512,
        'head': 'naive_head',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # anchor box
        'anchor_size': [[16, 16], [32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # matcher
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1500,
        'warmup_factor': 0.00066667,
        'epoch': {
            '1x': {'max_epoch': 12, 
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24, 
                    'lr_epoch': [16, 22], 
                    'multi_scale': [480, 544, 608, 672, 736, 800]},
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [480, 544, 608, 672, 736, 800]},
        },
    },

    'yolof101': {
        # input
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': {
            '1x':[{'name': 'ToTensor'},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'RandomShift', 'max_shift': 32},
                    {'name': 'Resize'},
                    {'name': 'Normalize'},
                    {'name': 'PadImage'}],

            '2x':[{'name': 'ToTensor'},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'RandomShift', 'max_shift': 32},
                    {'name': 'Resize'},
                    {'name': 'Normalize'},
                    {'name': 'PadImage'}],

            '3x':[{'name': 'DistortTransform'},
                    {'name': 'ToTensor'},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'RandomShift', 'max_shift': 32},
                    {'name': 'RandomSizeCrop'},                 
                    {'name': 'Resize'},
                    {'name': 'Normalize'},
                    {'name': 'PadImage'}]},
        # model
        'backbone': 'resnet101',
        'norm_type': 'FrozeBN',
        'stride': 32,
        'act_type': 'relu',
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [2, 4, 6, 8],
        'expand_ratio': 0.25,
        # head
        'head_dim': 512,
        'head': 'naive_head',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # anchor box
        'anchor_size': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # matcher
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1500,
        'warmup_factor': 0.00066667,
        'epoch': {
            '1x': {'max_epoch': 12, 
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24, 
                    'lr_epoch': [16, 22], 
                    'multi_scale': [480, 544, 608, 672, 736, 800]},
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [480, 544, 608, 672, 736, 800]},
        },
    },

    'yolof101-DC5': {
        # input
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': {
            '1x':[{'name': 'ToTensor'},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'RandomShift', 'max_shift': 32},
                    {'name': 'Resize'},
                    {'name': 'Normalize'},
                    {'name': 'PadImage'}],

            '2x':[{'name': 'ToTensor'},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'RandomShift', 'max_shift': 32},
                    {'name': 'Resize'},
                    {'name': 'Normalize'},
                    {'name': 'PadImage'}],

            '3x':[{'name': 'DistortTransform'},
                    {'name': 'ToTensor'},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'RandomShift', 'max_shift': 32},
                    {'name': 'RandomSizeCrop'},                 
                    {'name': 'Resize'},
                    {'name': 'Normalize'},
                    {'name': 'PadImage'}]},
        # model
        'backbone': 'resnet101-d',
        'norm_type': 'FrozeBN',
        'stride': 16,
        'act_type': 'relu',
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [4, 8, 12, 16],
        'expand_ratio': 0.25,
        # head
        'head_dim': 512,
        'head': 'naive_head',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # anchor box
        'anchor_size': [[16, 16], [32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # matcher
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1500,
        'warmup_factor': 0.00066667,
        'epoch': {
            '1x': {'max_epoch': 12, 
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24, 
                    'lr_epoch': [16, 22], 
                    'multi_scale': [480, 544, 608, 672, 736, 800]},
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [480, 544, 608, 672, 736, 800]},
        },
    },

    'yolof53-DC5': {
        # input
        'format': 'BGR',
        'pixel_mean': [0.406, 0.456, 0.485],
        'pixel_std': [1.0, 1.0, 1.0],
        'transforms': {
            '1x':[{'name': 'ToTensor'},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'RandomShift', 'max_shift': 32},
                    {'name': 'Resize'},
                    {'name': 'Normalize'},
                    {'name': 'PadImage'}],

            '2x':[{'name': 'ToTensor'},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'RandomShift', 'max_shift': 32},
                    {'name': 'Resize'},
                    {'name': 'Normalize'},
                    {'name': 'PadImage'}],

            '3x':[{'name': 'DistortTransform'},
                    {'name': 'ToTensor'},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'RandomShift', 'max_shift': 32},
                    {'name': 'RandomSizeCrop'},                 
                    {'name': 'Resize'},
                    {'name': 'Normalize'},
                    {'name': 'PadImage'}],

            '9x':[{'name': 'DistortTransform'},
                    {'name': 'ToTensor'},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'RandomShift', 'max_shift': 32},
                    {'name': 'RandomSizeCrop'},                 
                    {'name': 'Resize'},
                    {'name': 'Normalize'},
                    {'name': 'PadImage'}]},
        # model
        'backbone': 'cspdarknet53-d',
        'norm_type': 'FrozeBN',
        'stride': 16,
        'act_type': 'lrelu',
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [4, 8, 12, 16],
        'expand_ratio': 0.25,
        # head
        'head_dim': 512,
        'head': 'naive_head',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # anchor box
        'anchor_size': [[16, 16], [32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # matcher
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1500,
        'warmup_factor': 0.00066667,
        'epoch': {
            '1x': {'max_epoch': 12, 
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24, 
                    'lr_epoch': [16, 22], 
                    'multi_scale': [448, 512, 544, 576, 608, 640]},
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [448, 512, 544, 576, 608, 640]},
            '9x': {'max_epoch': 108, 
                    'lr_epoch': [72, 99], 
                    'multi_scale': [448, 512, 544, 576, 608, 640]},
        },
    },
}