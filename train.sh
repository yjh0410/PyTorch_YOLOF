python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v fcos-r50 \
        --batch_size 8 \
        --schedule 1x \
        --eval_epoch 1 \
        --grad_clip_norm 4.0 \
