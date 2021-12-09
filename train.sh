python train.py \
        --cuda \
        -d voc \
        -v yolof_r50_C5_1x \
        --batch_size 16 \
        --no_warmup \
        --accumulate 4