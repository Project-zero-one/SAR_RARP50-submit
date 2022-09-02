#!/usr/bin/env bash
base_dir="/mnt/cloudy_z/src/atsushi/mmsegmentation"
save_dir="/mnt/cloudy_z/result/Nerve/mmseg"

trial_list=(
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_selfsup.py"
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_gaussianCE_nerve_small.py"
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_positive_simsiam_positive.py"
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_negative_simsiam_positive.py"
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_negative_simsiam_negative.py"
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
        --gpu-ids 0
        # --resume-from $work_dir/latest.pth
done
