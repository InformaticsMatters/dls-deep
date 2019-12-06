ARG from_image="informaticsmatters-deep-rdkit-centos7"
ARG from_tag="latest"
FROM ${from_image}:${from_tag}

# CUDA Toolkit (10.2) ---------------------------------------------------------

RUN yum-config-manager --add-repo http://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo && \
    yum clean all && \
    yum -y install cuda-toolkit-10-2

ENV CUDA_ROOT_DIR /usr/local/cuda-10.2
ENV CUDA_BIN_DIR ${CUDA_ROOT_DIR}/bin
ENV CUDACXX ${CUDA_BIN_DIR}/nvcc

# A label for this package
LABEL tool.cuda="v10.2"
