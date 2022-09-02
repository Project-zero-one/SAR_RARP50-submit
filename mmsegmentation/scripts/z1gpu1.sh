#!/usr/bin/env bash
base_dir="/mnt/cloudy_z/src/atsushi/mmsegmentation"
save_dir="/mnt/cloudy_z/result/Nerve/mmseg"

trial_list=(
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_smoothCE_nerve_small.py"
    # "sem_fpn/nerve/simsiam_scene1346/fpn_efficientnetv2_l_512x256_20epoch_nerve_positive_simsiam_scene1346_20000frs_cv5.py"
    # "sem_fpn/nerve/simsiam_scene1346/fpn_efficientnetv2_l_512x256_20epoch_nerve_positive_simsiam_scene1346_20000frs_cv45.py"
    # "sem_fpn/nerve/simsiam_scene1346/fpn_efficientnetv2_l_512x256_20epoch_nerve_positive_simsiam_scene1346_20000frs_cv345.py"
    "sem_fpn/nerve/simsiam_scene1346/fpn_efficientnetv2_l_512x256_20epoch_nerve_positive_simsiam_scene1346_0frs_cv5.py"
    "sem_fpn/nerve/simsiam_scene1346/fpn_efficientnetv2_l_512x256_20epoch_nerve_positive_simsiam_scene1346_0frs_cv45.py"
    "sem_fpn/nerve/simsiam_scene1346/fpn_efficientnetv2_l_512x256_20epoch_nerve_positive_simsiam_scene1346_0frs_cv345.py"
    "sem_fpn/nerve/simsiam_scene1346/fpn_efficientnetv2_l_512x256_20epoch_nerve_random_cv5.py"
    "sem_fpn/nerve/simsiam_scene1346/fpn_efficientnetv2_l_512x256_20epoch_nerve_random_cv45.py"
    "sem_fpn/nerve/simsiam_scene1346/fpn_efficientnetv2_l_512x256_20epoch_nerve_random_cv345.py"
    "sem_fpn/nerve/simsiam_scene1346/fpn_efficientnetv2_l_512x256_20epoch_nerve_random.py"
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
done
