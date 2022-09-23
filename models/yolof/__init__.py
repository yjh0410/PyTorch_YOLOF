import torch
from .yolof import YOLOF


# build YOLOF detector
def build_model(args, 
                cfg, 
                device, 
                num_classes=80, 
                trainable=False, 
                pretrained=None,
                eval_mode=False):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))

    if trainable:
        conf_thresh = cfg['conf_thresh_val']
        nms_thresh = cfg['nms_thresh_val']
    else:
        if eval_mode:
            conf_thresh = cfg['conf_thresh_val']
            nms_thresh = cfg['nms_thresh_val']
        else:
            conf_thresh = cfg['conf_thresh']
            nms_thresh = cfg['nms_thresh']

    model = YOLOF(cfg=cfg,
                  device=device, 
                  num_classes=num_classes, 
                  trainable=trainable,
                  conf_thresh=conf_thresh,
                  nms_thresh=nms_thresh,
                  topk=args.topk)

    # Load pretrained weight
    if pretrained is not None:
        print('Loading pretrained weight ...')
        checkpoint = torch.load(pretrained, map_location='cpu')
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
                print(k)

        model.load_state_dict(checkpoint_state_dict, strict=False)
                        
    return model
