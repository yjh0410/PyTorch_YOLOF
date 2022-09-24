from .resnet import build_resnet


def build_backbone(cfg, pretrained=False):
    print('==============================')
    print('Backbone: {}'.format(cfg['backbone'].upper()))
    print('--pretrained: {}'.format(pretrained))

    if cfg['backbone'] in ['resnet18', 'resnet50', 'resnet101']:
        model, feat_dim = build_resnet(
            model_name=cfg['backbone'], 
            pretrained=pretrained,
            norm_type=cfg['bk_norm_type'],
            res5_dilation=cfg['res5_dilation']
            )

    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
