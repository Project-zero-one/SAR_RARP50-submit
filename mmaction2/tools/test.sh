# python test.py \
#     /mnt/cloudy_z/src/yharai/mmaction2/work_dir/sar-rarp50/swin_tiny_patch244_window877_kinetics400_1k/cv1/cv1.py \
#     /mnt/cloudy_z/src/yharai/mmaction2/work_dir/sar-rarp50/swin_tiny_patch244_window877_kinetics400_1k/cv1/epoch_30.pth \
#     --out /mnt/cloudy_z/src/yharai/mmaction2/work_dir/sar-rarp50/swin_tiny_patch244_window877_kinetics400_1k/cv1//result.json \
#     --eval top_k_accuracy

python ./sar-rarp50/test.py \
    /mnt/cloudy_z/result/SAR-RARP50/ActionRecognition/VideoClassification/slowfast_r50_4x16x1_256e_kinetics400_rgb/coloraug/cv3/cv3.py \
    /mnt/cloudy_z/result/SAR-RARP50/ActionRecognition/VideoClassification/slowfast_r50_4x16x1_256e_kinetics400_rgb/coloraug/cv3/epoch_10.pth \
    --out /mnt/cloudy_z/result/SAR-RARP50/ActionRecognition/VideoClassification/slowfast_r50_4x16x1_256e_kinetics400_rgb/coloraug/cv3/result.json \
    --eval top_k_accuracy

# python test.py \
#     /mnt/cloudy_z/result/SAR-RARP50/ActionRecognition/VideoClassification/slowfast_r50_4x16x1_256e_kinetics400_rgb/coloraug/cv4/cv4.py \
#     /mnt/cloudy_z/result/SAR-RARP50/ActionRecognition/VideoClassification/slowfast_r50_4x16x1_256e_kinetics400_rgb/coloraug/cv4/epoch_10.pth \
#     --out /mnt/cloudy_z/result/SAR-RARP50/ActionRecognition/VideoClassification/slowfast_r50_4x16x1_256e_kinetics400_rgb/coloraug/cv4/result.json \
#     --eval top_k_accuracy

# python test.py \
#     /mnt/cloudy_z/result/SAR-RARP50/ActionRecognition/VideoClassification/slowfast_r50_4x16x1_256e_kinetics400_rgb/coloraug/cv5/cv5.py \
#     /mnt/cloudy_z/result/SAR-RARP50/ActionRecognition/VideoClassification/slowfast_r50_4x16x1_256e_kinetics400_rgb/coloraug/cv5/epoch_10.pth \
#     --out /mnt/cloudy_z/result/SAR-RARP50/ActionRecognition/VideoClassification/slowfast_r50_4x16x1_256e_kinetics400_rgb/coloraug/cv5/result.json \
#     --eval top_k_accuracy

# python test.py \
#     /mnt/cloudy_z/result/SAR-RARP50/ActionRecognition/VideoClassification/slowfast_r50_4x16x1_256e_kinetics400_rgb/coloraug_class_weights/cv1/cv1_coloraug_class_weights.py\
#     /mnt/cloudy_z/result/SAR-RARP50/ActionRecognition/VideoClassification/slowfast_r50_4x16x1_256e_kinetics400_rgb/coloraug_class_weights/cv1/epoch_10.pth \
#     --out /mnt/cloudy_z/result/SAR-RARP50/ActionRecognition/VideoClassification/slowfast_r50_4x16x1_256e_kinetics400_rgb/coloraug_class_weights/cv1/result.json \
#     --eval top_k_accuracy