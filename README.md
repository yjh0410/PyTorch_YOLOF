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

# See positive sample
You can run following command to visualize positiva sample:
```Shell
python train.py \
        -d voc \
        --batch_size 2 \
        --root path/to/your/dataset \
        --vis_targets
```

# My Ablation Studies

## image mask
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

<tr><th align="left" bgcolor=#f8f8f8> no bug </th><td bgcolor=white>   </td><td bgcolor=white>   </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

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

We ignore the negative samples whose IoU are higher the ignore threshold (igt).

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> Method </th><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> no igt </th><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> igt=0.7 </th><td bgcolor=white>   </td><td bgcolor=white>   </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

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
python test.py -d [select a dataset: voc or coco] \
               --cuda \
               -v [select a model] \
               --weight [ Please input the path to model dir. ] \
               --img_size 800 \
               --root path/to/dataset/ \
               --show
```

You can run the above command to visualize the detection results on the dataset.
