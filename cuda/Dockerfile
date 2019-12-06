ARG base_image="informaticsmatters-deep-rdkit-centos7:latest"
FROM ${base_image}

LABEL maintainer="achristie@informaticsmatters.com"

# CUDA Toolkit (10.2.89) ------------------------------------------------------

WORKDIR /home/cuda
RUN wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-rhel7-10-2-local-10.2.89-440.33.01-1.0-1.x86_64.rpm 2> /dev/null && \
    rpm -i cuda-repo-rhel7-10-2-local-10.2.89-440.33.01-1.0-1.x86_64.rpm && \
    yum clean all && \
    yum -y install cuda-toolkit-10-2 && \
    rm cuda-repo-rhel7-10-2-local-10.2.89-440.33.01-1.0-1.x86_64.rpm

ENV CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.2
ENV CUDA_TOOLKIT_BIN_DIR /usr/local/cuda-10.2/bin
ENV CUDACXX ${CUDA_TOOLKIT_BIN_DIR}/nvcc

# A label for this package
LABEL tool.cuda="v10.2.89"
