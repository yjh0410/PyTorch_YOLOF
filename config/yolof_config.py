# YOLOF config


yolof_config = {
    'yolof-r18': {
        # ----------------- PreProcess -----------------
        'train_min_size': 800,
        'train_max_size': 1333,
        'test_min_size': 800,
        'test_max_size': 1333,
        'format': 'RGB',
        'pixel_mean': [123.675, 116.28, 103.53],
        'pixel_std': [58.395, 57.12, 57.375],
        'min_box_size': 8,
        'mosaic': False,
        'transforms': [{'name': 'RandomHorizontalFlip'},
                       {'name': 'RandomShift', 'max_shift': 32},
                       {'name': 'ToTensor'},
                       {'name': 'Resize'},
                       {'name': 'Normalize'}],
        # ----------------- Network Parameters -----------------
        ## Backbone
        'backbone': 'resnet18',
        'res5_dilation': False,
        'stride': 32,
        'bk_act_type': 'relu',
        'bk_norm_type': 'FrozeBN',
        ## Neck: DilatedEncoder
        'dilation_list': [2, 4, 6, 8],
        'encoder_dim': 512,
        'expand_ratio': 0.25,
        'encoder_act_type': 'relu',
        'encoder_norm_type': 'BN',
        ## Head
        'num_cls_heads': 2,
        'num_reg_heads': 4,
        'decoder_act_type': 'relu',
        'decoder_norm_type': 'BN',
        # ----------------- PostProcess -----------------
        'conf_thresh': 0.1,
        'nms_thresh': 0.5,
        'conf_thresh_val': 0.05,
        'nms_thresh_val': 0.6,
        # ----------------- Anchor box Configuration -----------------
        'anchor_size': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # ----------------- Label Assignment -----------------
        ## UniformMatcher
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,
        # ----------------- Loss Configuration-----------------
        ## Loss hyper-parameters
        'alpha': 0.25,
        'gamma': 2.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        # ----------------- Training Configuration -----------------
        ## optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'base_lr': 0.12 / 64,
        'bk_lr_ratio': 1.0 / 3.0,
        ## Warmup
        'warmup': 'linear',
        'wp_iter': 1500,
        'warmup_factor': 0.00066667,
        ## Epoch
        'epoch': {
            '1x': {'max_epoch': 12, 
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24, 
                    'lr_epoch': [16, 22], 
                    'multi_scale': [640, 672, 704, 736, 768, 800]},
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [640, 672, 704, 736, 768, 800]},
        },
    },

    'yolof-r50': {
        # ----------------- PreProcess -----------------
        'train_min_size': 800,
        'train_max_size': 1333,
        'test_min_size': 800,
        'test_max_size': 1333,
        'format': 'RGB',
        'pixel_mean': [123.675, 116.28, 103.53],
        'pixel_std': [58.395, 57.12, 57.375],
        'min_box_size': 8,
        'mosaic': False,
        'transforms': [{'name': 'RandomHorizontalFlip'},
                       {'name': 'RandomShift', 'max_shift': 32},
                       {'name': 'ToTensor'},
                       {'name': 'Resize'},
                       {'name': 'Normalize'}],
        # ----------------- Network Parameters -----------------
        ## Backbone
        'backbone': 'resnet50',
        'res5_dilation': False,
        'stride': 32,
        'bk_act_type': 'relu',
        'bk_norm_type': 'FrozeBN',
        ## Neck: DilatedEncoder
        'dilation_list': [2, 4, 6, 8],
        'encoder_dim': 512,
        'expand_ratio': 0.25,
        'encoder_act_type': 'relu',
        'encoder_norm_type': 'BN',
        ## Head
        'num_cls_heads': 2,
        'num_reg_heads': 4,
        'decoder_act_type': 'relu',
        'decoder_norm_type': 'BN',
        # ----------------- PostProcess -----------------
        'conf_thresh': 0.1,
        'nms_thresh': 0.5,
        'conf_thresh_val': 0.05,
        'nms_thresh_val': 0.6,
        # ----------------- Anchor box Configuration -----------------
        'anchor_size': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # ----------------- Label Assignment -----------------
        ## UniformMatcher
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,
        # ----------------- Loss Configuration-----------------
        ## Loss hyper-parameters
        'alpha': 0.25,
        'gamma': 2.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        # ----------------- Training Configuration -----------------
        ## optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'base_lr': 0.12 / 64,
        'bk_lr_ratio': 1.0 / 3.0,
        ## Warmup
        'warmup': 'linear',
        'wp_iter': 1500,
        'warmup_factor': 0.00066667,
        ## Epoch
        'epoch': {
            '1x': {'max_epoch': 12, 
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24, 
                    'lr_epoch': [16, 22], 
                    'multi_scale': [640, 672, 704, 736, 768, 800]},
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [640, 672, 704, 736, 768, 800]},
        },
    },

    'yolof-r50-DC5': {
        # ----------------- PreProcess -----------------
        'train_min_size': 800,
        'train_max_size': 1333,
        'test_min_size': 800,
        'test_max_size': 1333,
        'format': 'RGB',
        'pixel_mean': [123.675, 116.28, 103.53],
        'pixel_std': [58.395, 57.12, 57.375],
        'min_box_size': 8,
        'mosaic': False,
        'transforms': [{'name': 'RandomHorizontalFlip'},
                       {'name': 'RandomShift', 'max_shift': 32},
                       {'name': 'ToTensor'},
                       {'name': 'Resize'},
                       {'name': 'Normalize'}],
        # ----------------- Network Parameters -----------------
        ## Backbone
        'backbone': 'resnet50',
        'res5_dilation': True,
        'stride': 16,
        'bk_act_type': 'relu',
        'bk_norm_type': 'FrozeBN',
        ## Neck: DilatedEncoder
        'dilation_list': [4, 8, 12, 16],
        'encoder_dim': 512,
        'expand_ratio': 0.25,
        'encoder_act_type': 'relu',
        'encoder_norm_type': 'BN',
        ## Head
        'num_cls_heads': 2,
        'num_reg_heads': 4,
        'decoder_act_type': 'relu',
        'decoder_norm_type': 'BN',
        # ----------------- PostProcess -----------------
        'conf_thresh': 0.1,
        'nms_thresh': 0.5,
        'conf_thresh_val': 0.05,
        'nms_thresh_val': 0.6,
        # ----------------- Anchor box Configuration -----------------
        'anchor_size': [[16, 16], [32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # ----------------- Label Assignment -----------------
        ## UniformMatcher
        'topk': 8,
        'iou_t': 0.1,
        'igt': 0.7,
        'ctr_clamp': 32,
        # ----------------- Loss Configuration-----------------
        ## Loss hyper-parameters
        'alpha': 0.25,
        'gamma': 2.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        # ----------------- Training Configuration -----------------
        ## optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'base_lr': 0.12 / 64,
        'bk_lr_ratio': 1.0 / 3.0,
        ## Warmup
        'warmup': 'linear',
        'wp_iter': 1500,
        'warmup_factor': 0.00066667,
        ## Epoch
        'epoch': {
            '1x': {'max_epoch': 12, 
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24, 
                    'lr_epoch': [16, 22], 
                    'multi_scale': [640, 672, 704, 736, 768, 800]},
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [640, 672, 704, 736, 768, 800]},
        },
    },

    'yolof-rt-r50': {
        # ----------------- PreProcess -----------------
        'train_min_size': 512,
        'train_max_size': 900,
        'test_min_size': 512,
        'test_max_size': 736,
        'format': 'RGB',
        'pixel_mean': [123.675, 116.28, 103.53],
        'pixel_std': [58.395, 57.12, 57.375],
        'min_box_size': 8,
        'mosaic': True,
        'transforms': [{'name': 'DistortTransform',
                        'hue': 0.1,
                        'saturation': 1.5,
                        'exposure': 1.5},
                        {'name': 'RandomHorizontalFlip'},
                        {'name': 'RandomShift', 'max_shift': 32},
                        {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                        {'name': 'ToTensor'},
                        {'name': 'Resize'},
                        {'name': 'Normalize'}],
        # ----------------- Network Parameters -----------------
        ## Backbone
        'backbone': 'resnet50',
        'res5_dilation': True,
        'stride': 16,
        'bk_act_type': 'relu',
        'bk_norm_type': 'FrozeBN',
        ## Neck: DilatedEncoder
        'dilation_list': [1, 2, 4, 6, 8],
        'encoder_dim': 512,
        'expand_ratio': 0.25,
        'encoder_act_type': 'relu',
        'encoder_norm_type': 'BN',
        ## Head
        'num_cls_heads': 2,
        'num_reg_heads': 4,
        'decoder_act_type': 'relu',
        'decoder_norm_type': 'BN',
        # ----------------- PostProcess -----------------
        'conf_thresh': 0.1,
        'nms_thresh': 0.5,
        'conf_thresh_val': 0.05,
        'nms_thresh_val': 0.6,
        # ----------------- Anchor box Configuration -----------------
        'anchor_size': [[16, 16], [32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # ----------------- Label Assignment -----------------
        ## UniformMatcher
        'topk': 8,
        'iou_t': 0.1,
        'igt': 0.7,
        'ctr_clamp': 32,
        # ----------------- Loss Configuration-----------------
        ## Loss hyper-parameters
        'alpha': 0.25,
        'gamma': 2.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        # ----------------- Training Configuration -----------------
        ## optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'base_lr': 0.12 / 64,
        'bk_lr_ratio': 1.0 / 3.0,
        ## Warmup
        'warmup': 'linear',
        'wp_iter': 1500,
        'warmup_factor': 0.00066667,
        ## Epoch
        'epoch': {
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640]},
        },
    },

}