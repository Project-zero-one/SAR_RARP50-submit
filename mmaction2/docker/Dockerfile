ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"
ARG MMCV="1.4.2"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 8.0 7.0+PTX"
# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV http_proxy "http://gw.east.ncc.go.jp:8080"
ENV https_proxy "http://gw.east.ncc.go.jp:8080"
ENV HTTP_PROXY "http://gw.east.ncc.go.jp:8080"
ENV HTTPS_PROXY "http://gw.east.ncc.go.jp:8080"
RUN apt-key adv --keyserver-option http-proxy="http://gw.east.ncc.go.jp:8080" --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install mmcv-full
ARG PYTORCH
ARG CUDA
ARG MMCV
RUN ["/bin/bash", "-c", "pip install mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]
# RUN pip install mmcv-full==latest -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

# Install MMAction2
# RUN conda clean --all
# RUN git clone https://github.com/open-mmlab/mmaction2.git /mmaction2
# WORKDIR /mmaction2
# RUN mkdir -p /mmaction2/data
# ENV FORCE_CUDA="1"
# RUN pip install cython --no-cache-dir
# RUN pip install --no-cache-dir -e .
