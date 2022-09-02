# !/usr/bin/env bash
save_dir="/mnt/cloudy_z/result/Nerve/mmseg"
work_dirs=(
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small"
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_simclr"
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_mocov2"
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_byol"
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_densecl"
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_simsiam"
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_positive_simsiam_positive"
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_negative_simsiam_positive"
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_negative_simsiam_negative"
    # "sem_fpn/nerve/simsiam_scene1346/fpn_efficientnetv2_l_512x256_20epoch_nerve_positive_simsiam_scene1346_0frs"
    # "sem_fpn/nerve/simsiam_scene1346/fpn_efficientnetv2_l_512x256_20epoch_nerve_positive_simsiam_scene1346_5000frs"
    # "sem_fpn/nerve/simsiam_scene1346/fpn_efficientnetv2_l_512x256_20epoch_nerve_positive_simsiam_scene1346_10000frs"
    # "sem_fpn/nerve/simsiam_scene1346/fpn_efficientnetv2_l_512x256_20epoch_nerve_positive_simsiam_scene1346_20000frs"
    # "sem_fpn/nerve/simsiam_scene1346/fpn_efficientnetv2_l_512x256_20epoch_nerve_positive_simsiam_scene1346_52000frs"
    # "sem_fpn/nerve/simsiam_scene1346/fpn_efficientnetv2_l_512x256_20epoch_nerve_positive_simsiam_scene1346_105000frs"
    # "sem_fpn/nerve/simsiam_scene1346/fpn_efficientnetv2_l_512x256_20epoch_nerve_positive_simsiam_scene1346_210000frs"
    "sem_fpn/nerve/simsiam_scene1346/fpn_efficientnetv2_l_512x256_20epoch_nerve_positive_simsiam_scene1346_10case_900frs"
    "sem_fpn/nerve/simsiam_scene1346/fpn_efficientnetv2_l_512x256_20epoch_nerve_positive_simsiam_scene1346_50case_4500frs"
)
GPUS=1

for w_dir in ${work_dirs[@]}; do
    echo "001113212.mp4: ${w_dir}"
    python tools/predict_on_video.py \
    --work-dir $save_dir/${w_dir} \
    --gpus $GPUS \
    --video-path "/mnt/data_src/S_HAR/001113212(case50).mp4" \
    --start-frame 24799 \
    --end-frame 26599

    # echo "001122562.mp4: ${w_dir}"
    # python tools/predict_on_video.py \
    # --work-dir $save_dir/${w_dir} \
    # --gpus $GPUS \
    # --video-path "/mnt/data_src/LAP-LAR/001122562.mp4" \
    # --start-frame 47013 \
    # --end-frame 48813

    # echo "001111376.mp4: ${w_dir}"
    # python tools/predict_on_video.py \
    # --work-dir $save_dir/${w_dir} \
    # --gpus $GPUS\
    # --video-path "/mnt/data_src/S_HAR/001111376.mp4" \
    # --start-frame 177075 \
    # --end-frame 178875

    # echo "001113472.mp4: ${w_dir}"
    # python tools/predict_on_video.py \
    # --work-dir $save_dir/${w_dir} \
    # --gpus $GPUS \
    # --video-path "/mnt/data_src/S_HAR/001113472(case193).mp4" \
    # --start-frame 75794 \
    # --end-frame 77594

    # echo "001111508.mp4: ${w_dir}"
    # python tools/predict_on_video.py \
    # --work-dir $save_dir/${w_dir} \
    # --gpus $GPUS \
    # --video-path "/mnt/data_src/S_HAR/001111508.mp4" \
    # --start-frame 140400 \
    # --end-frame 141600

    echo "001112057.mp4: ${w_dir}"
    python tools/predict_on_video.py \
    --work-dir $save_dir/${w_dir} \
    --gpus $GPUS \
    --video-path "/mnt/data_src/S_HAR/001112057.mp4" \
    --start-frame 33400 \
    --end-frame 34600

done
