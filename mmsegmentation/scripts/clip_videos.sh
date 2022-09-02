#!/usr/bin/env bash
save_dir="/mnt/cloudy_z/result/Nerve/mmseg/videos/raw_video"

## lumb
# ffmpeg \
#     -i /mnt/data_src/S_HAR/001111385.mp4 \
#     -vf trim=start_frame=60283:end_frame=62083,setpts=PTS-STARTPTS \
#     -an $save_dir/001111385_60283_62083.mp4

# ffmpeg \
#     -i /mnt/data_src/S_HAR/001111863.mp4 \
#     -vf trim=start_frame=62822:end_frame=64622,setpts=PTS-STARTPTS \
#     -an $save_dir/001111863_62822_64622.mp4

# ffmpeg \
#     -i /mnt/data_src/S_HAR/001111597.mp4 \
#     -vf trim=start_frame=104960:end_frame=106760,setpts=PTS-STARTPTS \
#     -an $save_dir/001111597_104960_106760.mp4

# ffmpeg \
#     -i "/mnt/data_src/S_HAR/001113793(case138).mp4" \
#     -vf trim=start_frame=51319:end_frame=53119,setpts=PTS-STARTPTS \
#     -an $save_dir/"001113793(case138)_51319_53119.mp4"

# ffmpeg \
#     -i "/mnt/data_src/S_HAR/001111486.mp4" \
#     -vf trim=start_frame=87100:end_frame=88900,setpts=PTS-STARTPTS \
#     -an $save_dir/"001111486_87100_88900.mp4"

# ffmpeg \
#     -i "/mnt/data_src/S_HAR/001113483(case215).mp4" \
#     -vf trim=start_frame=52100:end_frame=53900,setpts=PTS-STARTPTS \
#     -an $save_dir/"001113483(case215)_52100_53900.mp4"

# ffmpeg \
#     -i "/mnt/data_src/S_HAR/001111508.mp4" \
#     -vf trim=start_frame=89847:end_frame=91647,setpts=PTS-STARTPTS \
#     -an $save_dir/"001111508_89847_91647.mp4"

## hypo
# ffmpeg \
#     -i "/mnt/data_src/S_HAR/001113212(case50).mp4" \
#     -vf trim=start_frame=24799:end_frame=26599,setpts=PTS-STARTPTS \
#     -an $save_dir/"001113212(case50)_24799_26599.mp4"

# ffmpeg \
#     -i  "/mnt/data_src/LAP-LAR/001122562.mp4" \
#     -vf trim=start_frame=47013:end_frame=48813,setpts=PTS-STARTPTS \
#     -an $save_dir/"001122562_47013_48813.mp4"

# ffmpeg \
#     -i "/mnt/data_src/S_HAR/001111376.mp4" \
#     -vf trim=start_frame=177075:end_frame=178875,setpts=PTS-STARTPTS \
#     -an $save_dir/"001111376_177075_178875.mp4"

# ffmpeg \
#     -i "/mnt/data_src/S_HAR/001113472(case193).mp4" \
#     -vf trim=start_frame=75794:end_frame=77594,setpts=PTS-STARTPTS \
#     -an $save_dir/"001113472(case193)_75794_77594.mp4"

# ffmpeg \
#     -i "/mnt/data_src/S_HAR/001111508.mp4" \
#     -vf trim=start_frame=140400:end_frame=141600,setpts=PTS-STARTPTS \
#     -an $save_dir/"001111508_140400_141600.mp4"

# ffmpeg \
#     -i "/mnt/data_src/S_HAR/001112057.mp4" \
#     -vf trim=start_frame=33400:end_frame=34600,setpts=PTS-STARTPTS \
#     -an $save_dir/"001112057_33400_34600.mp4"
