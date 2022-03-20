python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v yolof50-DC5 \
        -lr 0.03 \
        -lr_bk 0.01 \
        --batch_size 16 \
        --train_min_size 640 \
        --train_max_size 640 \
        --val_min_size 640 \
        --val_max_size 640 \
        --schedule 3x \
        --grad_clip_norm 4.0 \

# Default:
# --train_min_size 800
# --train_max_size 1333
# --val_min_size 800
# --val_max_size 1333
