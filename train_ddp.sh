# 2 GPUs
python -m torch.distributed.run --nproc_per_node=2 train.py \
                                                    --cuda \
                                                    -dist \
                                                    -d coco \
                                                    --root /mnt/share/ssd2/dataset/ \
                                                    -v yolof-r50 \
                                                    --batch_size 8 \
                                                    --grad_clip_norm 4.0 \
                                                    --num_workers 4 \
                                                    --schedule 1x \
