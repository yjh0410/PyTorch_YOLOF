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

| Model                                     |  scale     |   mAP   |  FPS  |
|-------------------------------------------|------------|---------|-------|
| YOLOF_R_18_C5_1x                          |  800,1333  |         |       |
| YOLOF_R_50_C5_1x                          |  800,1333  |         |       |
| YOLOF_R_50_DC5_1x                         |  800,1333  | -       |   -   |
| YOLOF_R_101_C5_1x                         |  800,1333  | -       |   -   |
| YOLOF_R_101_DC5_1x                        |  800,1333  | -       |   -   |
| YOLOF_CSP_D_53_DC5_3x                     |  608,608   |         |       |
| YOLOF_CSP_D_53_DC5_9x                     |  608,608   | -       |   -   |

Limited by my computing resources, I can only provide the results and weights of `YOLOF_R_18_C5_1x`, `YOLOF_R_50_C5_1x` and `YOLOF_CSP_D53_DC5_3x`. If you have sufficient computing resources, you can try other backbones. I would be very grateful if you would release your trained weights and results for this project.

# Weight
- YOLOF_R_18_C5_1x: Coming soon ...
- YOLOF_R_50_C5_1x: Coming soon ...
- YOLOF_CSP_D_53_DC5_3x: Coming soon ...

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

## Test
```Shell
python test.py -d coco \
               --cuda \
               -v yolof50 \
               --weight path/to/weight \
               --min_size 800 \
               --max_size 1333 \
               --root path/to/dataset/ \
               --show
```
