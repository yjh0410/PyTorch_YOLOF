from .resnet import build_resnet


def build_backbone(model_name='resnet50-d', 
                   pretrained=False, 
                   norm_type='BN',
                   res5_dilation=False):
    print('==============================')
    print('Backbone: {}'.format(model_name.upper()))
    print('--pretrained: {}'.format(pretrained))

    if 'resnet' in model_name:
        model, feat_dim = build_resnet(
            model_name=model_name, 
            pretrained=pretrained,
            norm_type=norm_type,
            res5_dilation=res5_dilation
            )

    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
