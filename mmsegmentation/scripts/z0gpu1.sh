#!/usr/bin/env bash
base_dir="/mnt/cloudy_z/src/atsushi/mmsegmentation"
save_dir="/mnt/cloudy_z/result/SAR-RARP50/Segmentation"

trial_list=(
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_densecl.py"
    # "sem_fpn/fpn_efficientnetv2_l_512x256_20epoch_nerve_small_simclr.py"
    "senformer/sarrarp50/senformer_fpnt_swin_base_256x512/cv1.py"
)

GPU=1

for trial in ${trial_list[@]}; do
    config=$base_dir/"configs"/$trial
    work_dir=$save_dir/"${trial%.*}"
    echo "trial: $config"
    env CUDA_DEVICE_ORDER=PCI_BUS_ID LANG=C.UTF-8 CUDA_VISIBLE_DEVICES=$GPU \
    python tools/train.py \
        $config \
        --work-dir $work_dir \
        --deterministic \
        --seed 3407  #--resume-from $work_dir/latest.pth
    # plot loss and accuracy
    json_path_list=($(ls -t $work_dir/*.json))
    log_json=($(basename ${json_path_list[0]}))
    echo "log: $log_json"
    # train loss
    python tools/analysis_tools/analyze_logs.py \
        plot_curve \
        $work_dir/$log_json \
        --out $work_dir/loss.png
    # val IoU
    python tools/analysis_tools/analyze_logs.py \
        plot_curve \
        $work_dir/$log_json \
        --keys mIoU \
        --out $work_dir/mIoU.png
done
