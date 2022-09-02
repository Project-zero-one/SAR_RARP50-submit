#!/usr/bin/env bash
# cv_num=(
#     "cv1"
#     # "cv2"
#     # "cv3"
#     # "cv4"
#     # "cv5"
# )

# for cv in ${cv_num[@]}; do
#     python tools/predict_on_image_sarrarp50.py \
#         --config /mnt/cloudy_z/result/SAR-RARP50/Segmentation/segformer/segformer_mit-b3_512x256_30epoch/${cv}/config.py \
#         --checkpoint /mnt/cloudy_z/result/SAR-RARP50/Segmentation/segformer/segformer_mit-b3_512x256_30epoch/${cv}/latest.pth \
#         --gpus 0 \
#         --dataset-dir /mnt/data1/input/SAR-RARP50/20220723/${cv}/valid \
#         --save-dir /mnt/cloudy_z/result/SAR-RARP50/Segmentation/segformer/segformer_mit-b3_512x256_30epoch/${cv}/pred_val
# done


# cv_num=(
#     "cv1"
#     "cv2"
#     "cv3"
#     "cv4"
#     "cv5"
# )

# for cv in ${cv_num[@]}; do
#     python tools/predict_on_image_sarrarp50.py \
#         --config /mnt/cloudy_z/result/SAR-RARP50/Segmentation/swin/sarrarp50/upernet_swin_base_patch4_window7_256x512_focal_dice_loss/${cv}/config.py \
#         --checkpoint /mnt/cloudy_z/result/SAR-RARP50/Segmentation/swin/sarrarp50/upernet_swin_base_patch4_window7_256x512_focal_dice_loss/${cv}/latest.pth \
#         --gpus 0 \
#         --dataset-dir /mnt/data1/input/SAR-RARP50/20220723/${cv}/valid \
#         --save-dir /mnt/cloudy_z/result/SAR-RARP50/Segmentation/swin/sarrarp50/upernet_swin_base_patch4_window7_256x512_focal_dice_loss/${cv}/pred_val
# done


# for cv in ${cv_num[@]}; do
#     python tools/predict_on_image_sarrarp50.py \
#         --config /mnt/cloudy_z/result/SAR-RARP50/Segmentation/ocrnet/sarrarp50/ocrnet_hr48_512x256_20e_focal_dice_loss/${cv}/config.py \
#         --checkpoint /mnt/cloudy_z/result/SAR-RARP50/Segmentation/ocrnet/sarrarp50/ocrnet_hr48_512x256_20e_focal_dice_loss/${cv}/latest.pth \
#         --gpus 0 \
#         --dataset-dir /mnt/data1/input/SAR-RARP50/20220723/${cv}/valid \
#         --save-dir /mnt/cloudy_z/result/SAR-RARP50/Segmentation/ocrnet/sarrarp50/ocrnet_hr48_512x256_20e_focal_dice_loss/${cv}/pred_val
# done

#!/usr/bin/env bash
# cv_num=(
    # "cv1"
    # "cv2"
    # "cv3"
    # "cv4"
    # "cv5"
# )

# for cv in ${cv_num[@]}; do
#     python tools/predict_on_image_sarrarp50.py \
#         --config /mnt/cloudy_z/result/SAR-RARP50/Segmentation/segformer/segformer_mit-b3_512x256_30epoch/${cv}/config.py \
#         --checkpoint /mnt/cloudy_z/result/SAR-RARP50/Segmentation/segformer/segformer_mit-b3_512x256_30epoch/${cv}/latest.pth \
#         --gpus 0 \
#         --dataset-dir /mnt/data1/input/SAR-RARP50/20220723/${cv}/valid \
#         --save-dir /mnt/cloudy_z/result/SAR-RARP50/Segmentation/segformer/segformer_mit-b3_512x256_30epoch/${cv}/pred_val_scores
# done


# for cv in ${cv_num[@]}; do
#     python tools/predict_on_image_sarrarp50.py \
#         --config /mnt/cloudy_z/result/SAR-RARP50/Segmentation/swin/sarrarp50/upernet_swin_base_patch4_window7_256x512_focal_dice_loss/${cv}/config.py \
#         --checkpoint /mnt/cloudy_z/result/SAR-RARP50/Segmentation/swin/sarrarp50/upernet_swin_base_patch4_window7_256x512_focal_dice_loss/${cv}/latest.pth \
#         --gpus 0 \
#         --dataset-dir /mnt/data1/input/SAR-RARP50/20220723/${cv}/valid \
#         --save-dir /mnt/cloudy_z/result/SAR-RARP50/Segmentation/swin/sarrarp50/upernet_swin_base_patch4_window7_256x512_focal_dice_loss/${cv}/pred_val_scores
# done


# for cv in ${cv_num[@]}; do
#     python tools/predict_on_image_sarrarp50.py \
#         --config /mnt/cloudy_z/result/SAR-RARP50/Segmentation/ocrnet/sarrarp50/ocrnet_hr48_512x256_20e_focal_dice_loss/${cv}/config.py \
#         --checkpoint /mnt/cloudy_z/result/SAR-RARP50/Segmentation/ocrnet/sarrarp50/ocrnet_hr48_512x256_20e_focal_dice_loss/${cv}/latest.pth \
#         --gpus 0 \
#         --dataset-dir /mnt/data1/input/SAR-RARP50/20220723/${cv}/valid \
#         --save-dir /mnt/cloudy_z/result/SAR-RARP50/Segmentation/ocrnet/sarrarp50/ocrnet_hr48_512x256_20e_focal_dice_loss/${cv}/pred_val_scores
# done


python tools/predict_on_image_sarrarp50.py \
    --config /mnt/cloudy_z/src/yishikawa/MICCAI/2022/SAR_RARP50-evaluation/weights/mmseg/ensemble/segformer/cv1/config.py \
    --checkpoint /mnt/cloudy_z/src/yishikawa/MICCAI/2022/SAR_RARP50-evaluation/weights/mmseg/ensemble/segformer/cv1/latest.pth \
    --gpus 0 \
    --dataset-dir /mnt/data1/input/SAR-RARP50/for_submit_test/annotation \
    --save-dir /mnt/data1/input/SAR-RARP50/for_submit_test/predictions_segmentation3
