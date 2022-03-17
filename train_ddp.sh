# 8 GPUs
python -m torch.distributed.run --nproc_per_node=8 train.py \
                                                    --cuda \
                                                    -dist \
                                                    --num_gpu8 \
                                                    -d coco \
                                                    --root /mnt/share/ssd2/dataset/ \
                                                    -v yolof50 \
                                                    -lr 0.12 \
                                                    -lr_bk 0.04 \
                                                    --batch_size 8 \
                                                    --train_min_size 800 \
                                                    --train_max_size 1333 \
                                                    --val_min_size 800 \
                                                    --val_max_size 1333 \
                                                    --grad_clip_norm 4.0 \
                                                    --num_workers 4 \
                                                    --schedule 1x \
                                                    # --mosaic
