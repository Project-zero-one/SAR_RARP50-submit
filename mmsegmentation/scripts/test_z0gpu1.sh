#!/usr/bin/env bash
# python tools/test.py \
#     /mnt/cloudy_z/src/atsushi/mmsegmentation/work_dirs/ocrnet_hr48_1280x720_40epoch_lumbar/ocrnet_hr48_1280x720_40epoch_lumbar.py \
#     /mnt/cloudy_z/src/atsushi/mmsegmentation/work_dirs/ocrnet_hr48_1280x720_40epoch_lumbar/epoch_6.pth \
#     --work-dir /mnt/cloudy_z/src/atsushi/mmsegmentation/work_dirs/ocrnet_hr48_1280x720_40epoch_lumbar \
#     --eval mFscore

# echo "fpn_efficientnetv2_l_512x256_20epoch_nerve_small/best_Fscore.Nerve_epoch_14.pth"
# python tools/test.py \
#     /mnt/cloudy_z/src/atsushi/mmsegmentation/configs/sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small.py \
#     /mnt/cloudy_z/src/atsushi/mmsegmentation/work_dirs/fpn_efficientnetv2_l_512x256_20epoch_nerve_small/best_Fscore.Nerve_epoch_14.pth \
#     --work-dir /mnt/cloudy_z/src/atsushi/mmsegmentation/work_dirs/fpn_efficientnetv2_l_512x256_20epoch_nerve_small \
#     --eval mFscore \
#     --gpu-ids 1
# echo "fpn_efficientnetv2_l_512x256_20epoch_nerve_small/latest.pth"
# python tools/test.py \
#     /mnt/cloudy_z/src/atsushi/mmsegmentation/configs/sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small.py \
#     /mnt/cloudy_z/src/atsushi/mmsegmentation/work_dirs/fpn_efficientnetv2_l_512x256_20epoch_nerve_small/latest.pth \
#     --work-dir /mnt/cloudy_z/src/atsushi/mmsegmentation/work_dirs/fpn_efficientnetv2_l_512x256_20epoch_nerve_small \
#     --eval mFscore \
#     --gpu-ids 1

# echo "deeplabv3plus_efficientnetv2_l_512x256_20epoch_nerve_small/best_Fscore.Nerve_epoch_16.pth"
# python tools/test.py \
#     /mnt/cloudy_z/src/atsushi/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_efficientnetv2_l_512x256_20epoch_nerve_small.py \
#     /mnt/cloudy_z/src/atsushi/mmsegmentation/work_dirs/deeplabv3plus_efficientnetv2_l_512x256_20epoch_nerve_small/best_Fscore.Nerve_epoch_16.pth \
#     --work-dir /mnt/cloudy_z/src/atsushi/mmsegmentation/work_dirs/deeplabv3plus_efficientnetv2_l_512x256_20epoch_nerve_small \
#     --eval mFscore \
#     --gpu-ids 1
# echo "deeplabv3plus_efficientnetv2_l_512x256_20epoch_nerve_small/latest.pth"
# python tools/test.py \
#     /mnt/cloudy_z/src/atsushi/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_efficientnetv2_l_512x256_20epoch_nerve_small.py \
#     /mnt/cloudy_z/src/atsushi/mmsegmentation/work_dirs/deeplabv3plus_efficientnetv2_l_512x256_20epoch_nerve_small/latest.pth \
#     --work-dir /mnt/cloudy_z/src/atsushi/mmsegmentation/work_dirs/deeplabv3plus_efficientnetv2_l_512x256_20epoch_nerve_small \
#     --eval mFscore \
#     --gpu-ids 1

# echo "ocrnet_deeplab_effcientnetv2_l_512x256_20epoch_nerve_small/best_Fscore.Nerve_epoch_9.pth"
# python tools/test.py \
#     /mnt/cloudy_z/src/atsushi/mmsegmentation/configs/ocrnet/ocrnet_deeplab_effcientnetv2_l_512x256_20epoch_nerve_small.py \
#     /mnt/cloudy_z/src/atsushi/mmsegmentation/work_dirs/ocrnet_deeplab_effcientnetv2_l_512x256_20epoch_nerve_small/best_Fscore.Nerve_epoch_9.pth \
#     --work-dir /mnt/cloudy_z/src/atsushi/mmsegmentation/work_dirs/ocrnet_deeplab_effcientnetv2_l_512x256_20epoch_nerve_small \
#     --eval mFscore \
#     --gpu-ids 1
# echo "ocrnet_deeplab_effcientnetv2_l_512x256_20epoch_nerve_small/latest.pth"
# python tools/test.py \
#     /mnt/cloudy_z/src/atsushi/mmsegmentation/configs/ocrnet/ocrnet_deeplab_effcientnetv2_l_512x256_20epoch_nerve_small.py \
#     /mnt/cloudy_z/src/atsushi/mmsegmentation/work_dirs/ocrnet_deeplab_effcientnetv2_l_512x256_20epoch_nerve_small/latest.pth \
#     --work-dir /mnt/cloudy_z/src/atsushi/mmsegmentation/work_dirs/ocrnet_deeplab_effcientnetv2_l_512x256_20epoch_nerve_small \
#     --eval mFscore \
#     --gpu-ids 1

# echo "ocrnet_deeplab_hr48_512x256_20epoch_nerve_small/best_Fscore.Nerve_epoch_9.pth"
# python tools/test.py \
#     /mnt/cloudy_z/src/atsushi/mmsegmentation/configs/ocrnet/ocrnet_deeplab_hr48_512x256_20epoch_nerve_small.py \
#     /mnt/cloudy_z/src/atsushi/mmsegmentation/work_dirs/ocrnet_deeplab_hr48_512x256_20epoch_nerve_small/best_Fscore.Nerve_epoch_9.pth \
#     --work-dir /mnt/cloudy_z/src/atsushi/mmsegmentation/work_dirs/ocrnet_deeplab_hr48_512x256_20epoch_nerve_small \
#     --eval mFscore \
#     --gpu-ids 1
# echo "ocrnet_deeplab_hr48_512x256_20epoch_nerve_small/latest.pth"
# python tools/test.py \
#     /mnt/cloudy_z/src/atsushi/mmsegmentation/configs/ocrnet/ocrnet_deeplab_hr48_512x256_20epoch_nerve_small.py \
#     /mnt/cloudy_z/src/atsushi/mmsegmentation/work_dirs/ocrnet_deeplab_hr48_512x256_20epoch_nerve_small/latest.pth \
#     --work-dir /mnt/cloudy_z/src/atsushi/mmsegmentation/work_dirs/ocrnet_deeplab_hr48_512x256_20epoch_nerve_small \
#     --eval mFscore \
#     --gpu-ids 1

# echo "ocrnet_hr48_512x256_20epoch_nerve_small_optuna/trial9/best_Fscore.Nerve_epoch_18.pth"
# python tools/test.py \
#     /mnt/cloudy_z/src/atsushi/mmsegmentation/configs/ocrnet/ocrnet_hr48_512x256_20epoch_nerve_small.py \
#     /mnt/cloudy_z/src/atsushi/mmsegmentation/work_dirs/ocrnet_hr48_512x256_20epoch_nerve_small_optuna/trial9/best_Fscore.Nerve_epoch_18.pth \
#     --work-dir /mnt/cloudy_z/src/atsushi/mmsegmentation/work_dirs/ocrnet_hr48_512x256_20epoch_nerve_small_optuna/trial9 \
#     --eval mFscore \
#     --gpu-ids 1
# echo "ocrnet_hr48_512x256_20epoch_nerve_small_optuna/trial9/latest.pth"
# python tools/test.py \
#     /mnt/cloudy_z/src/atsushi/mmsegmentation/configs/ocrnet/ocrnet_hr48_512x256_20epoch_nerve_small.py \
#     /mnt/cloudy_z/src/atsushi/mmsegmentation/work_dirs/ocrnet_hr48_512x256_20epoch_nerve_small_optuna/trial9/latest.pth \
#     --work-dir /mnt/cloudy_z/src/atsushi/mmsegmentation/work_dirs/ocrnet_hr48_512x256_20epoch_nerve_small_optuna/trial9 \
#     --eval mFscore \
#     --gpu-ids 1

    