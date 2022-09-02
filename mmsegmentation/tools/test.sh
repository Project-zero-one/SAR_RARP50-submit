#!/usr/bin/env bash
python tools/test.py \
    /mnt/cloudy_z/src/yishikawa/mmsegmentation/work_dirs/SAR-RARP50/segformer_mit-b3_512x256_30epoch_cv1/config.py \
    /mnt/cloudy_z/src/yishikawa/mmsegmentation/work_dirs/SAR-RARP50/segformer_mit-b3_512x256_30epoch_cv1/latest.pth \
    --eval mIoU \
    --gpu-id 2

# base_dir="/mnt/cloudy_z/result/Nerve/mmseg/sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_positive_simsiam_positive"
# python tools/test.py \
#     $base_dir/hypo_negative/config.py \
#     $base_dir/hypo_negative/best_0_Fscore.Nerve_epoch_14.pth \
#     --work-dir $base_dir/hypo_negative \
#     --eval mFscore \
#     --gpu-id 0