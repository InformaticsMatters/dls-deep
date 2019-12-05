FROM centos:centos7.7.1908

LABEL maintainer="achristie@informaticsmatters.com"

# The number of processors
# (for faster, multi-threaded make)
ARG N_PROC=4

# General dependencies --------------------------------------------------------
# Please keep this list (roughly) alphabetical
# so it's easy to check what we're installing
# when we want to add more.

RUN yum -y -q install epel-release && \
    yum -y -q groupinstall 'Development Tools' && \
    yum -y -q install \
        atlas-devel.x86_64 \
        atlas.x86_64 \
        atlas-static.x86_64 \
        bzip2-devel.x86_64 \
        eigen3-devel.noarch \
        gflags-devel.x86_64 \
        git \
        glog-devel.x86_64 \
        leveldb-devel.x86_64 \
        lmdb-devel.x86_64 \
        numpy.x86_64 \
        opencv-devel.x86_64 \
        perl-core \
        protobuf-compiler.x86_64 \
        protobuf-devel.x86_64 \
        python-devel.x86_64 \
        readline-devel.x86_64 \
        snappy-devel.x86_64 \
        sqlite-devel.x86_64 \
        wget \
        zlib-devel.x86_64

# gcc (8.x) -------------------------------------------------------------------
# Install a more recent gcc/development toolset.
# The 4.x verion that's part of the base OS is of little use to us.
# We then enable the desired collection in the shell for all users
# by writing a bash script in /etc/profile.d.
#
# This will fix it for a login shell but for RUN commands
# in Docker you will need to `source` the toolset for each RUN command
# where you need the latest gcc, e.g.: -
#
#   RUN source scl_source enable devtoolset-8 && \
#       gcc --version

RUN yum -y install centos-release-scl && \
    yum -y install devtoolset-8 && \
    echo 'source scl_source enable devtoolset-8' > /etc/profile.d/devtoolset-8.sh

# cmake (3.15.5) --------------------------------------------------------------
# Essential for OpenBabel and libmolgrid builds

WORKDIR /home/cmake
RUN source scl_source enable devtoolset-8 && \
    wget https://cmake.org/files/v3.15/cmake-3.15.5.tar.gz 2> /dev/null && \
    tar -xf cmake-3.* && \
    rm cmake-3.15.5.tar.gz && \
    cd cmake-3.* && \
    ./bootstrap --prefix=/usr/local && \
    make -j${N_PROC} && \
    make -j${N_PROC} install && \
    find . -name "*.o" -exec rm -rf {} \;

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
LABEL cuda="v10.2.89"

# Wrap-up ---------------------------------------------------------------------

ENV N_PROC ${N_PROC}

WORKDIR /
