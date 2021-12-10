from __future__ import division

import os
import argparse
import time
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.voc import VOCDetection
from data.coco import COCODataset
from config.yolof_config import yolof_config
from data.transforms import TrainTransforms, ValTransforms

from utils import distributed_utils
from utils.criterion import build_criterion
from utils.com_flops_params import FLOPs_and_Params
from utils.misc import detection_collate
from utils.vis import vis_data

from evaluator.coco_evaluator import COCOAPIEvaluator
from evaluator.voc_evaluator import VOCAPIEvaluator

from models.yolof import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOF Detection')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=800,
                        help='The shorter size of the input image')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--num_gpu', default=1, type=int, 
                        help='Number of GPUs to train')
    parser.add_argument('--eval_epoch', type=int,
                            default=2, help='interval between evaluations')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='Gamma update for SGD')

    # visualize
    parser.add_argument('--vis_data', action='store_true', default=False,
                        help='visualize input data.')
    parser.add_argument('--vis_targets', action='store_true', default=False,
                        help='visualize the targets.')
    parser.add_argument('--vis_anchors', action='store_true', default=False,
                        help='visualize anchor boxes.')
    # model
    parser.add_argument('-v', '--version', default='yolof_r50_C5_1x',
                        help='yolof_r50_C5_1x, yolof_r101_C5_1x')
    parser.add_argument('--norm', default='BN', type=str,
                        help='normalization layer')
    parser.add_argument('--conf_thresh', default=0.05, type=float,
                        help='NMS threshold')
    parser.add_argument('--nms_thresh', default=0.6, type=float,
                        help='NMS threshold')

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc, widerface, crowdhuman')
    
    # Loss
    parser.add_argument('--loss_cls_weight', default=1.0, type=float,
                        help='weight of cls loss')
    parser.add_argument('--loss_reg_weight', default=1.0, type=float,
                        help='weight of reg loss')
    # train trick
    parser.add_argument('--no_warmup', action='store_true', default=False,
                        help='do not use warmup')
    parser.add_argument('--wp_iter', type=int, default=500,
                        help='The upper bound of warm-up')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')
    parser.add_argument('--accumulate', type=int, default=1,
                        help='accumulate gradient')

    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--local_rank', type=int, default=0, 
                        help='local_rank')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # path to save model
    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # set distributed
    local_rank = 0
    if args.distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = torch.distributed.get_rank()
        print(local_rank)
        torch.cuda.set_device(local_rank)

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # YOLOF Config
    print('Model: ', args.version)
    cfg = yolof_config[args.version]

    # dataset and evaluator
    dataset, evaluator, num_classes = build_dataset(args, device)

    # dataloader
    dataloader = build_dataloader(args, dataset, detection_collate)

    # criterion
    criterion = build_criterion(args=args, device=device, cfg=cfg, num_classes=num_classes)
    
    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")

    # build model
    net = build_model(args=args, 
                      cfg=cfg,
                      device=device, 
                      num_classes=num_classes, 
                      trainable=True, 
                      post_process=False)
    model = net
    model = model.to(device).train()

    # compute FLOPs and Params
    if local_rank == 0:
        model.post_process = True
        model.eval()
        FLOPs_and_Params(model=model, size=args.img_size, device=device)
        model.post_process = False
        model.train()

    # DDP
    if args.distributed and args.num_gpu > 1:
        print('using DDP ...')
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
     
    # use tfboard
    tblogger = None
    if args.tfboard:
        print('use tensorboard ...')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
        log_path = os.path.join('log/', args.dataset, c_time)
        os.makedirs(log_path, exist_ok=True)

        tblogger = SummaryWriter(log_path)
    
    # optimizer setup
    tmp_lr = base_lr = cfg['lr']
    optimizer = optim.SGD(model.parameters(), 
                            lr=tmp_lr, 
                            momentum=0.9,
                            weight_decay=1e-4)

    # training configuration
    max_epoch = cfg['max_epoch']
    lr_epoch = cfg['lr_epoch']
    batch_size = args.batch_size
    epoch_size = len(dataset) // (batch_size * args.num_gpu)
    best_map = -1.
    warmup = not args.no_warmup

    t0 = time.time()
    # start training loop
    for epoch in range(args.start_epoch, max_epoch):
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)            

        # use step lr decay
        if epoch in lr_epoch:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)

        # train one epoch
        for iter_i, (images, targets, masks) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size
            # warmup
            if ni < args.wp_iter and warmup:
                nw = args.wp_iter
                tmp_lr = base_lr * pow(ni / nw, 4)
                set_lr(optimizer, tmp_lr)

            elif ni == args.wp_iter and warmup:
                # warmup is over
                warmup = False
                tmp_lr = base_lr
                set_lr(optimizer, tmp_lr)

            # visualize input data
            if args.vis_data:
                vis_data(images, targets, masks)

            # to device
            images = images.to(device)
            masks = masks.to(device)

            # inference
            outputs = model(images, mask=masks)

            # compute loss
            cls_loss, reg_loss, total_loss = criterion(anchor_boxes=net.anchor_boxes,
                                                       outputs=outputs,
                                                       targets=targets,
                                                       stride=net.stride,
                                                       images=images,
                                                       vis_labels=args.vis_targets)
            
            total_loss = total_loss / args.accumulate

            loss_dict = dict(
                cls_loss=cls_loss,
                reg_loss=reg_loss,
                total_loss=total_loss
            )
            loss_dict_reduced = distributed_utils.reduce_loss_dict(loss_dict)

            # check loss
            if torch.isnan(total_loss):
                continue

            # Backward and Optimize
            if ni % args.accumulate == 0:
                total_loss.backward()        
                optimizer.step()
                optimizer.zero_grad()

            # display
            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    tblogger.add_scalar('cls loss',  loss_dict_reduced['cls_loss'].item(),  ni)
                    tblogger.add_scalar('reg loss',  loss_dict_reduced['reg_loss'].item(),  ni)
                
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f][Loss: cls %.2f || reg %.2f || size (%d, %d) || time: %.2f]'
                        % (epoch+1, 
                           max_epoch, 
                           iter_i, 
                           epoch_size, 
                           tmp_lr,
                           loss_dict['cls_loss'].item(), 
                           loss_dict['reg_loss'].item(), 
                           images.size(-2), images.size(-1), 
                           t1-t0),
                        flush=True)

                t0 = time.time()

        # evaluation
        if (epoch + 1) % args.eval_epoch == 0 or (epoch + 1) == max_epoch:
            model_eval = model.module if args.distributed else model

            if evaluator is None:
                print('No evaluator ...')
                print('Saving state, epoch:', epoch + 1)
                torch.save(model_eval.state_dict(), os.path.join(path_to_save, 
                            args.version + '_' + repr(epoch + 1) + '.pth'))  
                print('Keep training ...')
            else:
                print('eval ...')

                # set eval mode
                model_eval.post_process = True
                model_eval.eval()

                if local_rank == 0:
                    # evaluate
                    evaluator.evaluate(model_eval)

                    cur_map = evaluator.map
                    if cur_map > best_map:
                        # update best-map
                        best_map = cur_map
                        # save model
                        print('Saving state, epoch:', epoch + 1)
                        torch.save(model_eval.state_dict(), os.path.join(path_to_save, 
                                    args.version + '_' + repr(epoch + 1) + '_' + str(round(best_map*100, 1)) + '.pth'))  
                    if args.tfboard:
                        if args.dataset == 'voc':
                            tblogger.add_scalar('07test/mAP', evaluator.map, epoch)
                        elif args.dataset == 'coco':
                            tblogger.add_scalar('val/AP50_95', evaluator.ap50_95, epoch)
                            tblogger.add_scalar('val/AP50', evaluator.ap50, epoch)

                if args.distributed:
                    # wait for all processes to synchronize
                    dist.barrier()

                # set train mode.
                model_eval.post_process = False
                model_eval.train()
    
    if args.tfboard:
        tblogger.close()


def build_dataset(args, device):
    if args.dataset == 'voc':
        data_dir = os.path.join(args.root, 'VOCdevkit')
        num_classes = 20
        max_size = int(round(1333 / 800 * args.img_size))
        dataset = VOCDetection(
                        data_dir=data_dir,
                        transform=TrainTransforms(args.img_size, max_size=max_size))

        evaluator = VOCAPIEvaluator(
                        data_dir=data_dir,
                        device=device,
                        transform=ValTransforms(args.img_size))

    elif args.dataset == 'coco':
        data_dir = os.path.join(args.root, 'COCO')
        num_classes = 80
        max_size = int(round(1333 / 800 * args.img_size))
        dataset = COCODataset(
                    data_dir=data_dir,
                    image_set='train2017',
                    transform=TrainTransforms(args.img_size, max_size=max_size))

        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        device=device,
                        transform=ValTransforms(args.img_size)
                        )
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    return dataset, evaluator, num_classes


def build_dataloader(args, dataset, collate_fn=None):
    # distributed
    if args.distributed and args.num_gpu > 1:
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
                        shuffle=False,
                        batch_size=args.batch_size, 
                        collate_fn=collate_fn,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )
    return dataloader


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
