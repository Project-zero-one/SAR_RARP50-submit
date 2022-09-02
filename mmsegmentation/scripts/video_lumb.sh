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
    echo "001111385.mp4: ${w_dir}"
    python tools/predict_on_video.py \
    --work-dir $save_dir/${w_dir} \
    --gpus $GPUS \
    --video-path /mnt/data_src/S_HAR/001111385.mp4 \
    --start-frame 60283 \
    --end-frame 62083 

    # echo "001111863.mp4: ${w_dir}"
    # python tools/predict_on_video.py \
    # --work-dir $save_dir/${w_dir} \
    # --gpus $GPUS \
    # --video-path /mnt/data_src/S_HAR/001111863.mp4 \
    # --start-frame 62822 \
    # --end-frame 64622

    echo "001111597.mp4: ${w_dir}"
    python tools/predict_on_video.py \
    --work-dir $save_dir/${w_dir} \
    --gpus $GPUS\
    --video-path /mnt/data_src/S_HAR/001111597.mp4 \
    --start-frame 104960 \
    --end-frame 106760

    # echo "001113793.mp4: ${w_dir}"
    # python tools/predict_on_video.py \
    # --work-dir $save_dir/${w_dir} \
    # --gpus $GPUS \
    # --video-path "/mnt/data_src/S_HAR/001113793(case138).mp4" \
    # --start-frame 51319 \
    # --end-frame 53119

    # echo "001111486.mp4: ${w_dir}"
    # python tools/predict_on_video.py \
    # --work-dir $save_dir/${w_dir} \
    # --gpus $GPUS \
    # --video-path "/mnt/data_src/S_HAR/001111486.mp4" \
    # --start-frame 87100 \
    # --end-frame 88900

    # echo "001113483.mp4: ${w_dir}"
    # python tools/predict_on_video.py \
    # --work-dir $save_dir/${w_dir} \
    # --gpus $GPUS \
    # --video-path "/mnt/data_src/S_HAR/001113483(case215).mp4" \
    # --start-frame 52100 \
    # --end-frame 53900

    # echo "001111508.mp4: ${w_dir}"
    # python tools/predict_on_video.py \
    # --work-dir $save_dir/${w_dir} \
    # --gpus $GPUS \
    # --video-path "/mnt/data_src/S_HAR/001111508.mp4" \
    # --start-frame 89847 \
    # --end-frame 91647
done
