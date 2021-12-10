python train.py \
        --cuda \
        -d coco \
        -v yolof_r50_C5_1x \
        --batch_size 8 \
        --img_size 800 \
        --lr 0.01 \
        --norm GN \
        --wp_iter 1500 \
        --accumulate 8