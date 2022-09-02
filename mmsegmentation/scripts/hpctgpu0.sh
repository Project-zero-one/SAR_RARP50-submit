#!/usr/bin/env bash
python tools/train.py \
    /mnt/cloudy_z/src/atsushi/mmsegmentation/configs/slowfast/deeplabv3plus_slowfast_r50_80k_hpct.py \
    --gpu-ids 0