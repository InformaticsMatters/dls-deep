ARG from_image="informaticsmatters-deep-rdkit-centos8"
ARG from_tag="latest"
FROM ${from_image}:${from_tag}

# CUDA Toolkit (10.2) ---------------------------------------------------------

RUN dnf config-manager --add-repo http://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo && \
    dnf clean all && \
    dnf -y --nogpg install cuda-toolkit-10-2

ENV CUDA_ROOT_DIR /usr/local/cuda-10.2
ENV CUDA_BIN_DIR ${CUDA_ROOT_DIR}/bin
ENV CUDACXX ${CUDA_BIN_DIR}/nvcc

# A label for this package
LABEL tool.cuda="v10.2"
