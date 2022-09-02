# Build docker image

```shell
cd /mnt/cloudy_z/src/yharai/mmaction2/docker
# docker build -t mmaction2:pytorch1.6.0-cuda10.1-cudnn7-mmcv1.3.16 .
```

```shell
docker build -t mmaction2:pytorch1.10.0-cuda11.3-cudnn8-mmcv1.4.2 .
```

To reflect host changes of mmaction2, comment out "git clone mmaction2" in Dockerfile.

# Run container

```shell
docker run \
    --gpus '"device=2"' \
    --shm-size=64gb \
    -itd \
    -p 58829:58829 \
    --name mmaction_yharai-gpu2 \
    -v /mnt:/mnt \
    mmaction2:pytorch1.10.0-cuda11.3-cudnn8-mmcv1.4.2
```

# Install mmaction2 to container
```shell
docker exec -it mmaction_yharai-gpu2 bash
cd /path/to/mmaction2
pip install --no-cache-dir -e .
```

# Options

## Denseflow

If you want to use tools/data/build_rawframes.py to create flow datasets, you may need to setup denseflow.
official install guide: https://github.com/open-mmlab/denseflow/blob/master/INSTALL.md

### Install wget
```shell
apt update
apt install wget
```

### Install cmake to build opencv
'apt install make' can not install make version > 3.10
```shell
Pip install cmake 
```
### Path to cmake
```shell
pip show cmake
export CMAKE_ROOT=/opt/conda/lib/python3.7/site-packages/cmake # set shown path
```

## Create flow datasets command
```shell
python ./data/build_rawframes.py INPUT_DATA_PATH OUTPUT_DATA_PATH --task flow --input-frames --out-format png --level 0 -o /mnt/cloudy_z/src/yharai/notebook/SAR-RARP50/testdataset/videos/video_1/flow
```

## Use Video Swin Transformer
Need to install apex
```shell
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```