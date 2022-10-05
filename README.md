# YOLOF: You Only Look At One-level Feature

This is a PyTorch version of YOLOF.

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

We suggest that PyTorch should be higher than 1.9.0 and Torchvision should be higher than 0.10.3. At least, please make sure your torch is version 1.x.

# Main results on COCO-val
- AP results of YOLOF

| Model                 |  scale     |  FPS  |   AP   |  AP50  | Weight |
|-----------------------|------------|-------|--------|--------|--------|
| YOLOF_R18_C5_1x       |  800,1333  |  100  |  32.6  |  51.3  | [github](https://github.com/yjh0410/PyTorch_YOLOF/releases/download/YOLOF-weight/yolof_r18_C5_1x_32.6.pth) |
| YOLOF_R50_C5_1x       |  800,1333  |  50   |  37.5  |  57.4  | [github](https://github.com/yjh0410/PyTorch_YOLOF/releases/download/YOLOF-weight/yolof-r50_C5_1x_37.5.pth) |
| YOLOF_R50_DC5_1x      |  800,1333  |  32   |  38.7  |  58.5  | [github](https://github.com/yjh0410/PyTorch_YOLOF/releases/download/YOLOF-weight/yolof-r50-DC5_1x_38.7.pth) |
| YOLOF_R101_C5_1x      |  800,1333  |       |        |        | [github]() |
| YOLOF_R101_DC5_1x     |  800,1333  |       |        |        | [github]() |
| YOLOF_R50-RT_3x       |  512,736   |       |  39.4  |  58.6  | [github]() |

Limited by my computing resources, I cannot train `YOLOF_R101_C5_1x`, `YOLOF_R101_DC5_1x`.
I would be very grateful if you used this project to train them and would like to share weight files.

- AP results of FCOS

| Model             |  scale     |  FPS  |   AP   |  AP50  | Weight |
|-------------------|------------|-------|--------|--------|--------|
| FCOS_R18_1x       |  800,1333  |  42   |  33.0  |  51.3  | [github](https://github.com/yjh0410/PyTorch_YOLOF/releases/download/YOLOF-weight/fcos-r18_1x_33.0.pth) |
| FCOS_R50_1x       |  800,1333  |  30   |  38.2  |  58.0  | [github]() |


- Visualization

(YOLOF_R50-C5_1x)

![image](./img_files/coco_samples.png)

# Train
- Single GPU

You can run the following command:
```Shell
python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v yolof-r50 \
        --batch_size 16 \
        --schedule 1x \
        --grad_clip_norm 4.0 \
```

or, you just run the script:
```Shell
sh train.sh
```

You can change the configurations of `train.sh`, according to your own situation.

- Multi GPUs

You can run the following command:
```Shell
# 2 GPUs
python -m torch.distributed.run --nproc_per_node=2 train.py \
                                                    --cuda \
                                                    -dist \
                                                    -d coco \
                                                    --root /mnt/share/ssd2/dataset/ \
                                                    -v yolof-r50 \
                                                    --batch_size 8 \
                                                    --grad_clip_norm 4.0 \
                                                    --num_workers 4 \
                                                    --schedule 1x \
```

or, you just run the script:
```Shell
sh train_ddp.sh
```

# Test
```Shell
python test.py -d coco \
               --cuda \
               -v yolof-r50 \
               --weight path/to/weight \
               --root path/to/dataset/ \
               --show
```

# Evaluate
```Shell
python eval.py -d coco-val \
               --cuda \
               -v yolof-r50 \
               --weight path/to/weight \
               --root path/to/dataset/ \
```

Our AP results of YOLOF-R50-C5-1x on COCO-val:

```Shell
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.375
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.574
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.399
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.185
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.421
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.522
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.312
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.507
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.552
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.332
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.625
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.739
```

# Demo
I have provide some images in `data/demo/images/`, so you can run following command to run a demo:

```Shell
python demo.py --mode image \
               --path_to_img data/demo/images/ \
               -v yolof-r50 \
               --cuda \
               --weight path/to/weight \
               --show
```

If you want run a demo of streaming video detection, you need to set `--mode` to `video`, and give the path to video `--path_to_vid`。

```Shell
python demo.py --mode video \
               --path_to_img data/demo/videos/your_video \
               -v yolof-r50 \
               --cuda \
               --weight path/to/weight \
               --show
```

If you want run video detection with your camera, you need to set `--mode` to `camera`。

```Shell
python demo.py --mode camera \
               -v yolof-r50 \
               --cuda \
               --weight path/to/weight \
               --show
```
