python train.py \
        --cuda \
        -d coco \
        -v yolof_r50_C5_1x \
        --batch_size 16 \
        --img_size 512 \
        --lr 0.03 \
        --lr_backbone 0.01 \
        --wp_iter 1500 \
        --accumulate 1