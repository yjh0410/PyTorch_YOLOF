python train.py \
        --cuda \
        -d voc \
        -v yolof_r50_C5_1x \
        --batch_size 2 \
        --img_size 800 \
        --norm GN \
        --accumulate 4