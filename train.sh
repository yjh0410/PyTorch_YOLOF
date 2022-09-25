python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v yolof-r50-DC5 \
        --batch_size 16 \
        --schedule 1x \
        --grad_clip_norm 4.0 \
