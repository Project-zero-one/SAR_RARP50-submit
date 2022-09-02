ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"
ARG MMCV="1.5.0"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 8.0 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV http_proxy "http://gw.east.ncc.go.jp:8080"
ENV https_proxy "http://gw.east.ncc.go.jp:8080"
ENV HTTP_PROXY "http://gw.east.ncc.go.jp:8080"
ENV HTTPS_PROXY "http://gw.east.ncc.go.jp:8080"
ENV CUDA_DEVICE_ORDER "PCI_BUS_ID"
ENV LANG "C.UTF-8"

RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y git vim ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-dev  \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN conda clean --all

# Install MMCV
ARG PYTORCH
ARG CUDA
ARG MMCV
RUN ["/bin/bash", "-c", "pip install mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]

ENV FORCE_CUDA="1"
# # Install MMSegmentation
# RUN git clone https://github.com/open-mmlab/mmsegmentation.git /mmsegmentation
# WORKDIR /mmsegmentation
# RUN pip install -r requirements.txt
# RUN pip install --no-cache-dir -e .
