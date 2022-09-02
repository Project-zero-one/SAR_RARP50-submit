#!/usr/bin/env bash
base_dir="/mnt/cloudy_z/result/Nerve/mmseg"
save_dir="/mnt/cloudy_z/result/Nerve/mmseg/videos/selfsup"

models=(
    "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_positive_simsiam_positive"
    "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_negative_simsiam_positive"
    "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_positive_simsiam_negative"
    "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_negative_simsiam_negative"
)

videos=(
    # lumb
    # "001111385_60283_62083.mp4"
    # "001111863_62822_64622.mp4"
    "001111597_104960_106760.mp4"
    # "001113793(case138)_51319_53119.mp4"
    # "001111486_87100_88900.mp4"
    # "001113483(case215)_52100_53900.mp4"
    # "001111508_89847_91647.mp4"

    # hypo
    # "001113212(case50)_24799_26599.mp4"
    # "001122562_47013_48813.mp4"
    # "001111376_177075_178875.mp4"
    # "001113472(case193)_75794_77594.mp4"
    # "001111508_140400_141600.mp4"
    "001112057_33400_34600.mp4"

    # "001123067_24500_26300.mp4"
    # "001141256_abd_39300_41100.mp4"
    # "001111881_113454_115254.mp4"
    # "001112057_98148_99948.mp4"
)

### 4モデル同時表示 ###
for v in ${videos[@]}; do
    ffmpeg \
    -i $base_dir/${models[0]}/prediction/${v} \
    -i $base_dir/${models[1]}/prediction/${v} \
    -i $base_dir/${models[2]}/prediction/${v} \
    -i $base_dir/${models[3]}/prediction/${v} \
    -filter_complex \
    "xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0" \
    $save_dir/${v%.*}"_4concat.mp4"
done

### 左上に元動画を載せる場合 ###
# for v in ${videos[@]}; do
#     ffmpeg \
#     -i "/mnt/cloudy_z/result/Nerve/mmseg/videos/raw_video/${v}" \
#     -i $base_dir/${models[0]}/prediction/${v} \
#     -i $base_dir/${models[1]}/prediction/${v} \
#     -i $base_dir/${models[2]}/prediction/${v} \
#     -filter_complex \
#     "xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0" \
#     $save_dir/${v%.*}"_4concat.mp4"
# done