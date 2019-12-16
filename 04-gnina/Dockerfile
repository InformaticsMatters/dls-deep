ARG from_image="informaticsmatters/deep-cuda-ubuntu"
ARG from_tag="latest"
FROM ${from_image}:${from_tag}

# System packages -------------------------------------------------------------

#RUN apt-get -y install \
#        git \
#        golang-goleveldb-dev \
#        libatlas-dev \
#        libatlas3-base \
#        libbz2-dev \
#        libeigen3-dev \
#        libgflags2v5 \
#        libgoogle-glog-dev \
#        liblmdb-dev \
#        libopencv-dev \
#        libprotobuf-dev \
#        libprotoc-dev \
#        libreadline-dev \
#        libsnappy-dev \
#        libsqlite3-dev \
#        perl \
#        protobuf-compiler \
#        wget \
#        zlib1g-dev


#        atlas-devel.x86_64 \
#        atlas.x86_64 \
#        bzip2-devel.x86_64 \
#        eigen3-devel.noarch \
#        gflags-devel.x86_64 \
#        git \
#        glog-devel.x86_64 \
#        leveldb-devel.x86_64 \
#        lmdb-devel.x86_64 \
#        opencv-devel.x86_64 \
#        perl-core \
#        protobuf-compiler.x86_64 \
#        protobuf-devel.x86_64 \
#        readline-devel.x86_64 \
#        snappy-devel.x86_64 \
#        sqlite-devel.x86_64 \
#        wget \
#        zlib-devel.x86_64

# libmolgrid (latest) ---------------------------------------------------------

ARG libmolgrid_version="a5bd251"

WORKDIR /root
RUN git clone https://github.com/gnina/libmolgrid.git

# Nobble test builds and testing because these appear not to build or run.
# By commenting-out the 'enable_testing' and 'add_subdirectory(test)' lines...
WORKDIR libmolgrid
RUN git checkout ${libmolgrid_version} && \
    sed -i 's/enable_testing/#enable_testing/g' CMakeLists.txt && \
    sed -ri 's/add_subdirectory[(]test[)]/#add_subdirectory(test)/g' CMakeLists.txt

WORKDIR build
RUN cmake .. \
        -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_ROOT_DIR} \
        -DPYTHON_EXECUTABLE=/usr/bin/python \
        -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m \
        -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so && \
     make -j${N_PROC} && \
     make -j${N_PROC} install && \
     cd .. && \
     rm -rf *

# A label for this package
LABEL libmolgrid="${libmolgrid_version}"

# gnina (latest) --------------------------------------------------------------

ARG gnina_version="16ce46d"

RUN apt-get -y install \
        libeigen3-dev \
        libgoogle-glog-dev \
        libprotobuf-dev \
        protobuf-compiler \
        libhdf5-serial-dev \
        libatlas-base-dev

WORKDIR /root
RUN git clone https://github.com/gnina/gnina.git

# Nobble test builds and testing because these appear not to build or run.
# By commenting-out the 'enable_testing' and 'add_subdirectory(test)' lines...
WORKDIR gnina
RUN git checkout ${gnina_version} && \
    sed -i 's/enable_testing/#enable_testing/g' CMakeLists.txt && \
    sed -ri 's/add_subdirectory[(]test[)]/#add_subdirectory(test)/g' CMakeLists.txt

WORKDIR build
RUN cmake .. \
        -DPYTHON_EXECUTABLE=/usr/bin/python \
        -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m \
        -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so \
#        -DAtlas_BLAS_LIBRARY=/usr/lib64/atlas/libtatlas.so \
#        -DAtlas_CBLAS_LIBRARY=/usr/lib64/atlas/libtatlas.so \
#        -DAtlas_LAPACK_LIBRARY=/usr/lib64/atlas/libtatlas.so \
#        -DHDF5_CXX_COMPILER_EXECUTABLE=/usr/bin/h5c++ \
#        -DHDF5_C_COMPILER_EXECUTABLE=/usr/bin/h5cc \
#        -DHDF5_DIFF_EXECUTABLE=/usr/bin/h5diff \
        -DOpenBabel3_INSTALL_PREFIX=/usr/local && \
    make -j${N_PROC} && \
    make -j${N_PROC} install && \
    make -j${N_PROC} pycaffe && \
    cd .. && \
    rm -rf *

# A label for this package
LABEL tool.gnina=${gnina_version}

WORKDIR /root
