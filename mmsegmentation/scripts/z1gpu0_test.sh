#!/usr/bin/env bash
base_dir="/mnt/cloudy_z/src/atsushi/mmsegmentation"
save_dir="/mnt/cloudy_z/result/Nerve/mmseg"

trial_list=(
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small.py"
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_simclr.py"
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_mocov2.py"
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_byol.py"
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_densecl.py"
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_simsiam.py"
)

for trial in ${trial_list[@]}; do
    config=$base_dir/"configs"/$trial
    work_dir=$save_dir/"${trial%.*}"
    echo "trial: $config"
    python tools/test.py \
        $config \
        $work_dir/model.pth \
        --work-dir $work_dir \
        --eval mFscore \
        --gpu-ids 0
done
