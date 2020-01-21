ARG from_image="informaticsmatters/deep-boost-ubuntu-1604"
ARG from_tag="latest"
FROM ${from_image}:${from_tag}

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

RUN apt-get -y --no-install-recommends install \
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
        -DOpenBabel3_INSTALL_PREFIX=/usr/local && \
    make -j${N_PROC} && \
    make -j${N_PROC} install && \
    make -j${N_PROC} pycaffe && \
    cd .. && \
    rm -rf *

ENV PYTHONPATH /usr/local/lib:$PYTHONPATH

# A label for this package
LABEL tool.gnina=${gnina_version}

WORKDIR /root
