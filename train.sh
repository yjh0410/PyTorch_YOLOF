python train.py \
        --cuda \
        -d voc \
        -v yolof_r50_C5_1x \
        --batch_size 8 \
        --img_size 512 \
        --norm GN \
        --accumulate 2