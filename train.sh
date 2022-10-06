python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v fcos-rt-r18 \
        --batch_size 16 \
        --schedule 4x \
        --eval_epoch 1 \
        --grad_clip_norm 4.0 \
