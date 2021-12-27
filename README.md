# PyTorch_YOLOF
A PyTorch version of You Only Look at One-level Feature object detector.

The input image must be resized to have their shorter side being 800 and their longer side less or equal to
1333. 

During reproducing the YOLOF, I found many tricks used in YOLOF but the baseline RetinaNet dosen't use those tricks.
For example, YOLOF takes advantage of RandomShift, CTR_CLAMP, large learning rate, big batchsize(like 64), negative prediction threshold. Is it really fair that YOLOF use these tricks to compare with RetinaNet?

In a other word, whether the YOLOF can still work without those tricks?

# Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n yolof python=3.6
```

- Then, activate the environment:
```Shell
conda activate yolof
```

- Requirements:
```Shell
pip install -r requirements.txt 
```
PyTorch >= 1.1.0 and Torchvision >= 0.3.0

# Visualize positive sample
You can run following command to visualize positiva sample:
```Shell
python train.py \
        -d voc \
        --batch_size 2 \
        --root path/to/your/dataset \
        --vis_targets
```

# My Ablation Studies

## Image mask
- Backbone: ResNet-50
- image size: shorter size = 800, longer size <= 1333
- Batch size: 16
- lr: 0.01
- lr of backbone: 0.01
- SGD with momentum 0.9 and weight decay 1e-4
- Matcher: IoU Top4 (Different from the official matcher that uses top4 of L1 distance.)
- epoch: 12 (1x schedule)
- lr decay: 8, 11
- augmentation: RandomFlip

We ignore the loss of samples who are not in image.

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> Method </th><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> w/o mask </th><td bgcolor=white> 28.3 </td><td bgcolor=white> 46.7 </td><td bgcolor=white> 28.9 </td><td bgcolor=white> 13.4 </td><td bgcolor=white> 33.4 </td><td bgcolor=white> 39.9 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> w mask   </th><td bgcolor=white> 28.4 </td><td bgcolor=white> 46.9 </td><td bgcolor=white> 29.1 </td><td bgcolor=white> 13.5 </td><td bgcolor=white> 33.5 </td><td bgcolor=white> 39.1 </td></tr>

<table><tbody>

## L1 Top4
- Backbone: ResNet-50
- image size: shorter size = 800, longer size <= 1333
- Batch size: 16
- lr: 0.01
- lr of backbone: 0.01
- SGD with momentum 0.9 and weight decay 1e-4
- epoch: 12 (1x schedule)
- lr decay: 8, 11
- augmentation: RandomFlip
- with image mask

IoU topk: We choose the topK of IoU between anchor boxes and labels as the positive samples.

L1 topk: We choose the topK of L1 distance between anchor boxes and labels as the positive samples.

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> Method </th><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> IoU Top4  </th><td bgcolor=white> 28.4 </td><td bgcolor=white> 46.9 </td><td bgcolor=white> 29.1 </td><td bgcolor=white> 13.5 </td><td bgcolor=white> 33.5 </td><td bgcolor=white> 39.1 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> L1 Top4 </th><td bgcolor=white> 28.6 </td><td bgcolor=white> 46.9 </td><td bgcolor=white> 29.4 </td><td bgcolor=white> 13.8 </td><td bgcolor=white> 34.0 </td><td bgcolor=white> 39.0 </td></tr>

<table><tbody>

## RandomShift Augmentation
- Backbone: ResNet-50
- image size: shorter size = 800, longer size <= 1333
- Batch size: 16
- lr: 0.01
- lr of backbone: 0.01
- SGD with momentum 0.9 and weight decay 1e-4
- Matcher: L1 Top4
- epoch: 12 (1x schedule)
- lr decay: 8, 11
- augmentation: RandomFlip
- with image mask

YOLOF takes advantage of RandomShift augmentation which is not used in RetinaNet.

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> Method </th><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> w/o RandomShift </th><td bgcolor=white> 28.6 </td><td bgcolor=white> 46.9 </td><td bgcolor=white> 29.4 </td><td bgcolor=white> 13.8 </td><td bgcolor=white> 34.0 </td><td bgcolor=white> 39.0 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> w/ RandomShift  </th><td bgcolor=white> 29.0 </td><td bgcolor=white> 47.3 </td><td bgcolor=white> 29.8 </td><td bgcolor=white> 14.2 </td><td bgcolor=white> 34.2 </td><td bgcolor=white> 38.9 </td></tr>

<table><tbody>


## Fix a bug in dataloader
- Backbone: ResNet-50
- image size: shorter size = 800, longer size <= 1333
- Batch size: 16
- lr: 0.01
- lr of backbone: 0.01
- SGD with momentum 0.9 and weight decay 1e-4
- Matcher: L1 Top4
- epoch: 12 (1x schedule)
- lr decay: 8, 11
- augmentation: RandomFlip + RandomShift
- with image mask

I fixed a bug in dataloader. Specifically, I set the `shuffle` in dataloader as `False` ...

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> Method </th><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> bug  </th><td bgcolor=white> 29.0 </td><td bgcolor=white> 47.3 </td><td bgcolor=white> 29.8 </td><td bgcolor=white> 14.2 </td><td bgcolor=white> 34.2 </td><td bgcolor=white> 38.9 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> no bug </th><td bgcolor=white> 30.1 </td><td bgcolor=white> 49.0 </td><td bgcolor=white> 31.0 </td><td bgcolor=white> 15.2 </td><td bgcolor=white> 36.3 </td><td bgcolor=white> 39.8 </td></tr>

<table><tbody>

## Ignore samples
- Backbone: ResNet-50
- image size: shorter size = 800, longer size <= 1333
- Batch size: 16
- lr: 0.01
- lr of backbone: 0.01
- SGD with momentum 0.9 and weight decay 1e-4
- Matcher: L1 Top4
- epoch: 12 (1x schedule)
- lr decay: 8, 11
- augmentation: RandomFlip + RandomShift
- with image mask

We ignore those negative samples whose IoU with labels are higher the ignore threshold (igt).

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> Method </th><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> no igt </th><td bgcolor=white> 30.1 </td><td bgcolor=white> 49.0 </td><td bgcolor=white> 31.0 </td><td bgcolor=white> 15.2 </td><td bgcolor=white> 36.3 </td><td bgcolor=white> 39.8 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> igt=0.7 </th><td bgcolor=white> 30.2 </td><td bgcolor=white> 49.3 </td><td bgcolor=white> 30.8 </td><td bgcolor=white> 15.5 </td><td bgcolor=white> 35.7 </td><td bgcolor=white> 41.2 </td></tr>

<table><tbody>

## Decode boxes
- Backbone: ResNet-50
- image size: shorter size = 800, longer size <= 1333
- Batch size: 16
- lr: 0.01
- lr of backbone: 0.01
- SGD with momentum 0.9 and weight decay 1e-4
- Matcher: L1 Top4
- epoch: 12 (1x schedule)
- lr decay: 8, 11
- augmentation: RandomFlip + RandomShift
- with image mask

Method-1: 
```Shell
x_c = x_anchor + t_x
y_c = y_anchor + t_y
```

Method-2: 
```Shell
x_c = x_anchor + t_x * w_anchor
y_c = y_anchor + t_y * h_anchor
```

The Method-2 is following the operation used in YOLOF.

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> Method </th><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Method-1 </th><td bgcolor=white> 30.2 </td><td bgcolor=white> 49.3 </td><td bgcolor=white> 30.8 </td><td bgcolor=white> 15.5 </td><td bgcolor=white> 35.7 </td><td bgcolor=white> 41.2 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Method-2 </th><td bgcolor=white> 30.1 </td><td bgcolor=white> 49.1 </td><td bgcolor=white> 30.9 </td><td bgcolor=white> 15.7 </td><td bgcolor=white> 35.4 </td><td bgcolor=white> 41.6 </td></tr>

<table><tbody>

## Scale loss by the number of total positive samples
- Backbone: ResNet-50
- image size: shorter size = 800, longer size <= 1333
- Batch size: 16
- lr: 0.01
- lr of backbone: 0.01
- SGD with momentum 0.9 and weight decay 1e-4
- Matcher: L1 Top4
- epoch: 12 (1x schedule)
- lr decay: 8, 11
- augmentation: RandomFlip + RandomShift
- with image mask
- Decode box: Method-2

Method-1:

In previous ablation studies, first we scale loss by the number of positive samples of each image, then we 
calculate the sum of total loss of the batch and scale the final loss by the batch size.
```Shell
# loss: (Tensor) [B, N,]
# num_pos: (Tensor) [B, N]
loss = loss.sum(-1) / num_pos.sum(-1)  # [B,]
loss = loss.sum() / batch_size
```

Method-2:

Now, we scale loss by the total positive samples of the batch.
```Shell
# loss: (Tensor) [B, N,]
# num_pos: (Tensor) [B, N]
loss = loss.sum() / num_pos.sum()
```

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> Scale loss </th><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Method-1 </th><td bgcolor=white> 30.1 </td><td bgcolor=white> 49.1 </td><td bgcolor=white> 30.9 </td><td bgcolor=white> 15.7 </td><td bgcolor=white> 35.4 </td><td bgcolor=white> 41.6 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Method-2 </th><td bgcolor=white> 31.4 </td><td bgcolor=white> 51.0 </td><td bgcolor=white> 32.4 </td><td bgcolor=white> 17.8 </td><td bgcolor=white> 37.8 </td><td bgcolor=white> 41.3 </td></tr>

<table><tbody>

## Prediction with 3 × 3 kernel size
- Backbone: ResNet-50
- image size: shorter size = 800, longer size <= 1333
- Batch size: 16
- lr: 0.01
- lr of backbone: 0.01
- SGD with momentum 0.9 and weight decay 1e-4
- Matcher: L1 Top4
- epoch: 12 (1x schedule)
- lr decay: 8, 11
- augmentation: RandomFlip + RandomShift
- with image mask
- Decode box: Method-2
- Scale loss: by number of total positive samples

In my previous YOLOF, I habitually set the kernel size in objectness, classification and regression to 1 × 1, just
as many detectors do like RetinaNet and YOLO.

After being reminded by others, I took a closer look at the source code of YOLOF again, and I found the kernel size
are all 3 × 3, not 1 × 1. Therefore, I reset the kernel size to 3 × 3.

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> Kernel size </th><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> 1 × 1 </th><td bgcolor=white> 31.4 </td><td bgcolor=white> 51.0 </td><td bgcolor=white> 32.4 </td><td bgcolor=white> 17.8 </td><td bgcolor=white> 37.8 </td><td bgcolor=white> 41.3 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> 3 × 3 </th><td bgcolor=white> 31.2 </td><td bgcolor=white> 50.8 </td><td bgcolor=white> 32.1 </td><td bgcolor=white> 17.3 </td><td bgcolor=white> 37.6 </td><td bgcolor=white> 40.9 </td></tr>

<table><tbody>

## Prediction with 3 × 3 kernel size
- Backbone: ResNet-50
- image size: shorter size = 800, longer size <= 1333
- Batch size: 16
- lr: 0.03
- lr of backbone: 0.01
- SGD with momentum 0.9 and weight decay 1e-4
- Matcher: L1 Top4
- epoch: 12 (1x schedule)
- lr decay: 8, 11
- augmentation: RandomFlip + RandomShift
- with image mask
- Decode box: Method-2
- Scale loss: by number of total positive samples

YOLOF uses large learning rate 0.12 (the learning rate of backbone is one-third of 0.12, that is, 0.04) with 
the 64 batch size. My YOLOF uses the 16 batch size since I have no many GPUs. Therefore, I should set the learning
rate as 0.03 (the learning rate of backbone is 0.01).

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> Lr </th><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> 0.01 </th><td bgcolor=white> 31.4 </td><td bgcolor=white> 51.0 </td><td bgcolor=white> 32.4 </td><td bgcolor=white> 17.8 </td><td bgcolor=white> 37.8 </td><td bgcolor=white> 41.3 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> 0.03 </th><td bgcolor=white> 31.4 </td><td bgcolor=white> 51.1 </td><td bgcolor=white> 32.6 </td><td bgcolor=white> 17.1 </td><td bgcolor=white> 37.8 </td><td bgcolor=white> 40.5 </td></tr>

<table><tbody>

It doesn't work.

## Accumulate 4 gradient
- Backbone: ResNet-50
- image size: shorter size = 800, longer size <= 1333
- Batch size: 16
- lr: 0.01
- lr of backbone: 0.01
- SGD with momentum 0.9 and weight decay 1e-4
- Matcher: L1 Top4
- epoch: 12 (1x schedule)
- lr decay: 8, 11
- augmentation: RandomFlip + RandomShift
- with image mask
- Decode box: Method-2
- Scale loss: by number of total positive samples

YOLOF uses 64 batch size. But I just can use 1 3090 GPU, so I can't set batch size as 64.
In this ablation study, I set `--accumulate` as 4, that is, the model accumulates 4 gradient
to update parameters.

If this trick still dosen't work, I have to give up. I have no any idea to optimize my YOLOF.

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> Accu </th><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> 1 </th><td bgcolor=white> 31.4 </td><td bgcolor=white> 51.0 </td><td bgcolor=white> 32.4 </td><td bgcolor=white> 17.8 </td><td bgcolor=white> 37.8 </td><td bgcolor=white> 41.3 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> 4 </th><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<table><tbody>


# Train
```Shell
sh train.sh
```

You can change the configurations of `train.sh`.

If you just want to check which anchor box is assigned to the positive sample, you can run:
```Shell
python train.py --cuda -d voc --batch_size 8 --vis_targets
```

According to your own situation, you can make necessary adjustments to the above run commands

## Test
```Shell
python test.py -d coco \
               --cuda \
               --weight path/to/weight \
               --min_size 800 \
               --max_size 1333 \
               --root path/to/dataset/ \
               --show
```

You can run the above command to visualize the detection results on the dataset.
