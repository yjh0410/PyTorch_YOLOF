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

| Model                 |  scale     |   AP   |  AP50  | Weight |  log  |
|-----------------------|------------|--------|--------|--------|-------|
| YOLOF_R18_C5_1x       |  800,1333  |  32.2  |  50.7  | [github](https://github.com/yjh0410/PyTorch_YOLOF/releases/download/YOLOF-weight/yolof_r18_C5_1x_32.2.pth) | [log]() |
| YOLOF_R50_C5_1x       |  800,1333  |  37.2  |  57.0  | [github]() | [log]() |
| YOLOF_R50_DC5_1x      |  800,1333  |        |        | [github]() | [log]() |
| YOLOF_R101_C5_1x      |  800,1333  |        |        | [github]() | [log]() |
| YOLOF_R101_DC5_1x     |  800,1333  |        |        | [github]() | [log]() |
| YOLOF_R50-RT_3x       |  800,1333  |        |        | [github]() | [log]() |

<!-- Limited by my computing resources, I cannot provide other weights files of `YOLOF_R_50_DC5_1x`, `YOLOF_R_101_C5_1x`and `YOLOF_R_101_DC5_1x`.
I would be very grateful if you used this project to train the above models and would like to open source the weights files.
 -->
# Train
## Single GPU
```Shell
sh train.sh
```

You can change the configurations of `train.sh`, according to your own situation.

## Multi GPUs
```Shell
sh train_ddp.sh
```

You can change the configurations of `train_ddp.sh`, according to your own situation.

# Test
```Shell
python test.py -d coco \
               --cuda \
               -v yolof-r50 \
               --weight path/to/weight \
               --root path/to/dataset/ \
               --show
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
