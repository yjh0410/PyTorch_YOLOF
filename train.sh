python train.py \
        --cuda \
        -d voc \
        --root /mnt/share/ssd2/dataset/ \
        -v yolof18 \
        --batch_size 16 \
        --schedule 1x \
        --grad_clip_norm 4.0 \
