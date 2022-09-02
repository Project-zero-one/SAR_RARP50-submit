#!/usr/bin/env bash
# python tools/optuna_study.py \
#     ./configs/segformer/segformer_mit-b0_512x256_10iter_lumbar_optuna.py \
#     --work-dir ./work_dirs/test \
#     --trials 3 \
#     --study-name trials \
#     --seed 3407 \
#     --deterministic \
#     --gpu-id 1

# python tools/optuna_study.py \
#     configs/sem_fpn/nerve/fpn_efficientnetv2_l_512x256_20epoch_nerve_optuna.py \
#     --work-dir /mnt/cloudy_z/result/Nerve/mmseg/sem_fpn/nerve/fpn_efficientnetv2_l_512x256_20epoch_nerve_optuna \
#     --trials 300 \
#     --study-name nerve_all_effv2 \
#     --seed 3407 \
#     --deterministic \
#     --gpu-id 2

# python tools/optuna_study.py \
#     configs/ocrnet/ocrnet_hr48_512x256_20epoch_nerve_optuna.py \
#     --work-dir /mnt/cloudy_z/result/Nerve/mmseg/ocrnet/ocrnet_hr48_512x256_20epoch_nerve_optuna \
#     --trials 300 \
#     --study-name nerve_all_ocr \
#     --seed 3407 \
#     --deterministic \
#     --gpu-id 3

python tools/optuna_study.py \
    /mnt/cloudy_z/src/yishikawa/mmsegmentation/configs/segformer/sar_rarp50/segformer_mit-b3_512x256_30epoch_cv1_optuna.py \
    --work-dir /mnt/cloudy_z/src/yishikawa/mmsegmentation/work_dirs/SAR-RARP50/segformer_mit-b3_512x256_30epoch_10cls_optuna/cv1 \
    --trials 30 \
    --study-name trials \
    --seed 3407 \
    --deterministic \
    --gpu-id 1

# python tools/optuna_study.py \
#     /mnt/cloudy_z/src/yishikawa/mmsegmentation/configs/segformer/sar_rarp50/segformer_mit-b3_512x256_30epoch_ftl_cv1_optuna.py \
#     --work-dir /mnt/cloudy_z/src/yishikawa/mmsegmentation/work_dirs/SAR-RARP50/segformer_mit-b3_512x256_30epoch_ftl_optuna/cv1 \
#     --trials 50 \
#     --study-name trials \
#     --seed 3407 \
#     --deterministic \
#     --gpu-id 0

    