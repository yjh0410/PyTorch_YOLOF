from torch import optim


def build_optimizer(cfg, model, base_lr=0.0, backbone_lr=0.0):
    print('==============================')
    print('Optimizer: {}'.format(cfg['optimizer']))
    print('--momentum: {}'.format(cfg['momentum']))
    print('--weight_decay: {}'.format(cfg['weight_decay']))

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": backbone_lr,
        },
    ]

    if cfg['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            params=param_dicts, 
            lr=base_lr,
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay']
            )

    elif cfg['optimizer'] == 'adam':
        optimizer = optim.Adam(
            params=param_dicts, 
            lr=base_lr,
            weight_decay=cfg['weight_decay']
            )
                                
    elif cfg['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            params=param_dicts, 
            lr=base_lr,
            weight_decay=cfg['weight_decay']
            )
                                
    return optimizer
