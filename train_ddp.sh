# 8 GPUs
python -m torch.distributed.launch --nproc_per_node=8 --use_env \
                    train.py \
                        --cuda \
                        -d coco \
                        -v yolof_r50_C5_1x \
                        --batch_size 2 \
                        --img_size 800 \
                        --lr 0.01 \
                        --norm GN \
                        --wp_iter 1500 \
                        --dist \
                        --num_gpu 8 \
                        --accumulate 32