# 8 GPUs
# I have no enough GPUs to debug this train script of DDP.
python -m torch.distributed.launch --nproc_per_node=8 --use_env \
                    train.py \
                        --cuda \
                        -d coco \
                        -v yolof_r50_C5_1x \
                        --batch_size 2 \
                        --img_size 800 \
                        --lr 0.01 \
                        --wp_iter 1500 \
                        --dist \
                        --sybn \
                        --num_gpu 8 \
                        --accumulate 1