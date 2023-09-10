python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v fcos-r18 \
        --batch_size 16 \
        --schedule 3x \
        --eval_epoch 2 \
        --grad_clip_norm 4.0 \
