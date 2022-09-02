#!/usr/bin/env bash
base_dir="/mnt/cloudy_z/src/atsushi/mmsegmentation"
save_dir="/mnt/cloudy_z/result/Nerve/mmseg"

trial_list=(
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_smoothCE_nerve_small.py"
)

for trial in ${trial_list[@]}; do
    config=$base_dir/"configs"/$trial
    work_dir=$save_dir/"${trial%.*}"
    echo "trial: $config"
    python tools/train.py \
        $config \
        --work-dir $work_dir \
        --deterministic \
        --seed 3407 \
        --gpu-ids 3
done
