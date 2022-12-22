from .yolof_config import yolof_config
from .fcos_config import fcos_config
from .retinanet_config import retinanet_config
from .yolof_lite_config import yolof_lite_config


def build_config(args):
    if args.version in ['yolof-r18', 'yolof-r50', 'yolof-r50-DC5',
                        'yolof-r101', 'yolof-r101-DC5', 'yolof-rt-r50']:
        return yolof_config[args.version]

    elif args.version in ['fcos-r18', 'fcos-r50', 'fcos-r101', 'fcos-rt-r18', 'fcos-rt-r50']:
        return fcos_config[args.version]

    elif args.version in ['retinanet-r18', 'retinanet-r50', 'retinanet-r101',
                          'retinanet-rt-r18', 'retinanet-rt-r50']:
        return retinanet_config[args.version]

    elif args.version in ['yolof-lite-r18', 'yolof-lite-r50', 'yolof-lite-r50-DC5',
                          'yolof-lite-r101', 'yolof-lite-r101-DC5', 'yolof-lite-rt-r50']:
        return yolof_lite_config[args.version]
 
