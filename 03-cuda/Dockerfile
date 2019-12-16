ARG from_image="informaticsmatters/deep-rdkit-ubuntu"
ARG from_tag="latest"
FROM ${from_image}:${from_tag}

# CUDA Toolkit (10.2) ---------------------------------------------------------

# We need add-apt-repository....
RUN apt-get -y install software-properties-common

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin 2> /dev/null && \
    mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/ /" && \
    apt-get update && \
    apt-get -y install cuda-toolkit-10-2

ENV CUDA_ROOT_DIR /usr/local/cuda-10.2
ENV CUDA_BIN_DIR ${CUDA_ROOT_DIR}/bin
ENV CUDACXX ${CUDA_BIN_DIR}/nvcc

# A label for this package
LABEL tool.cuda="v10.2"
