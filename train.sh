python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v yolof18 \
        -lr 0.03 \
        -lr_bk 0.01 \
        --batch_size 16 \
        --train_min_size 800 \
        --train_max_size 1333 \
        --val_min_size 800 \
        --val_max_size 1333 \
        --schedule 1x \
        --grad_clip_norm 4.0 \

# Default:
# --train_min_size 800
# --train_max_size 1333
# --val_min_size 800
# --val_max_size 1333
