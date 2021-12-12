# PyTorch_YOLOF
A PyTorch version of You Only Look at One-level Feature object detector.

The input image must be resized to have their shorter side being 800 and their longer side less or equal to
1333. 

During reproducing the YOLOF, I found many tricks used in YOLOF but the baseline RetinaNet dosen't use those tricks.
For example, YOLOF takes advantage of RandomShift, CTR_CLAMP, LR_BACKBONE, big batchsize(like 64), negative prediction threshold. Is it really fair that YOLO use these tricks to compare with RetinaNet?

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
