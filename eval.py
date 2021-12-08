import argparse
import os

import torch

from evaluator.voc_evaluator import VOCAPIEvaluator
from evaluator.coco_evaluator import COCOAPIEvaluator

from data.transforms import ValTransforms
from config import yolof_config

from utils.misc import TestTimeAugmentation

from models.yolof import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOF Detection')
    # basic
    parser.add_argument('-size', '--img_size', default=800, type=int,
                        help='the min size of input image')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')
    # model
    parser.add_argument('-v', '--version', default='yolof_r50_C5_1x',
                        help='yolof_r50_C5_1x, yolof_r101_C5_1x')
    parser.add_argument('--weight', default='weight/',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--conf_thresh', default=0.05, type=float,
                        help='NMS threshold')
    parser.add_argument('--nms_thresh', default=0.6, type=float,
                        help='NMS threshold')
    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc.')
    # TTA
    parser.add_argument('-tta', '--test_aug', action='store_true', default=False,
                        help='use test augmentation.')

    return parser.parse_args()



def voc_test(model, data_dir, device, img_size):
    evaluator = VOCAPIEvaluator(data_root=data_dir,
                                device=device,
                                transform=ValTransforms(img_size),
                                display=True)

    # VOC evaluation
    evaluator.evaluate(model)


def coco_test(model, data_dir, device, img_size, test=False):
    if test:
        # test-dev
        print('test on test-dev 2017')
        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        device=device,
                        testset=True,
                        transform=ValTransforms(img_size))

    else:
        # eval
        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        device=device,
                        testset=False,
                        transform=ValTransforms(img_size))

    # COCO evaluation
    evaluator.evaluate(model)


if __name__ == '__main__':
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # dataset
    if args.dataset == 'voc':
        print('eval on voc ...')
        num_classes = 20
        data_dir = os.path.join(args.root, 'VOCdevkit')
    elif args.dataset == 'coco-val':
        print('eval on coco-val ...')
        num_classes = 80
        data_dir = os.path.join(args.root, 'COCO')
    elif args.dataset == 'coco-test':
        print('eval on coco-test-dev ...')
        num_classes = 80
        data_dir = os.path.join(args.root, 'COCO')
    else:
        print('unknow dataset !! we only support voc, coco-val, coco-test !!!')
        exit(0)


    # YOLOF config
    print('Model: ', args.version)
    cfg = yolof_config[args.version]

    # build model
    model = build_model(args=args, 
                        device=device, 
                        num_classes=num_classes, 
                        trainable=False, 
                        anchor_size=cfg['anchor_size'], 
                        post_process=True)

    # load weight
    model.load_state_dict(torch.load(args.weight, map_location=device), strict=False)
    model = model.to(device).eval()
    print('Finished loading model!')

    # TTA
    test_aug = TestTimeAugmentation(num_classes=num_classes) if args.test_aug else None
    
    # evaluation
    with torch.no_grad():
        if args.dataset == 'voc':
            voc_test(model, data_dir, device, args.img_size)
        elif args.dataset == 'coco-val':
            coco_test(model, data_dir, device, args.img_size, test=False)
        elif args.dataset == 'coco-test':
            coco_test(model, data_dir, device, args.img_size, test=True)
