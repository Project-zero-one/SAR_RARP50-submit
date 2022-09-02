#!/usr/bin/env bash
python tools/optuna_study.py \
    /mnt/cloudy_z/src/atsushi/mmsegmentation/configs/resnest/deeplabv3plus_s101-d8_512x256_20epoch_nerve_small_optuna.py \
    --trials 100 \
    --study-name nerve_small_deeplab_resnest101 \
    --seed 3407 \
    --deterministic \
    --gpu-ids 0
