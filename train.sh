python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v retinanet-r18 \
        --batch_size 16 \
        --schedule 1x \
        --eval_epoch 2 \
        --grad_clip_norm 4.0 \
