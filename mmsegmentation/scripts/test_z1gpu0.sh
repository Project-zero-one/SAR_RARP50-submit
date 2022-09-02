#!/usr/bin/env bash
base_dir="/mnt/cloudy_z/src/atsushi/mmsegmentation"
save_dir="/mnt/cloudy_z/result/Nerve/mmseg"

# echo "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_optuna/trial42"
# python tools/test.py \
#     $base_dir/configs/sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small.py \
#     $save_dir/"sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_optuna/trial42"/model.pth \
#     --work-dir $save_dir/"sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_optuna/trial42" \
#     --eval mFscore \
#     --gpu-ids 0

# echo "ocrnet/ocrnet_hr48_512x256_20epoch_nerve_small_optuna/trial63"
# python tools/test.py \
#     $base_dir/configs/ocrnet/ocrnet_hr48_512x256_20epoch_nerve_small.py \
#     $save_dir/"ocrnet/ocrnet_hr48_512x256_20epoch_nerve_small_optuna/trial63"/model.pth \
#     --work-dir $save_dir/"ocrnet/ocrnet_hr48_512x256_20epoch_nerve_small_optuna/trial63" \
#     --eval mFscore \
#     --gpu-ids 0

# echo "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_simsiam"
# python tools/test.py \
#     $base_dir/configs/sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_simsiam.py \
#     $save_dir/"sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_simsiam"/model.pth \
#     --work-dir $save_dir/"sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_simsiam" \
#     --eval mFscore \
#     --gpu-ids 0