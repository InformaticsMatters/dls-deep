ARG base_image="informaticsmatters-deep-cuda-centos7:latest"
FROM ${base_image}

# libmolgrid (latest) ---------------------------------------------------------

ARG libmolgrid_version="a5bd251"

WORKDIR /home
RUN git clone https://github.com/gnina/libmolgrid.git

# Nobble test builds and testing because these appear not to build or run.
# By commenting-out the 'enable_testing' and 'add_subdirectory(test)' lines...
WORKDIR libmolgrid
RUN git checkout ${libmolgrid_version} && \
    sed -i 's/enable_testing/#enable_testing/g' CMakeLists.txt && \
    sed -ri 's/add_subdirectory[(]test[)]/#add_subdirectory(test)/g' CMakeLists.txt

WORKDIR build
RUN source scl_source enable devtoolset-8 && \
    cmake .. \
        -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR} \
        -DPYTHON_EXECUTABLE=/usr/bin/python3 \
        -DPYTHON_INCLUDE_DIR=/usr/local/include/python3.7m \
        -DPYTHON_LIBRARY=/usr/local/lib/libpython3.7m.so && \
     make -j${N_PROC} && \
     make -j${N_PROC} install

# A label for this package
LABEL libmolgrid="${libmolgrid_version}"

# gnina (latest) --------------------------------------------------------------

ARG gnina_version="16ce46d"

RUN yum -y install \
        hdf5-devel

WORKDIR /home
RUN git clone https://github.com/gnina/gnina.git

# Nobble test builds and testing because these appear not to build or run.
# By commenting-out the 'enable_testing' and 'add_subdirectory(test)' lines...
WORKDIR gnina
RUN git checkout ${gnina_version} && \
    sed -i 's/enable_testing/#enable_testing/g' CMakeLists.txt && \
    sed -ri 's/add_subdirectory[(]test[)]/#add_subdirectory(test)/g' CMakeLists.txt

WORKDIR build
RUN source scl_source enable devtoolset-8 && \
    cmake .. \
        -DPYTHON_EXECUTABLE=/usr/bin/python3 \
        -DPYTHON_INCLUDE_DIR=/usr/local/include/python3.7m \
        -DPYTHON_LIBRARY=/usr/local/lib/libpython3.7m.so \
        -DAtlas_BLAS_LIBRARY=/usr/lib64/atlas/libtatlas.so \
        -DAtlas_CBLAS_LIBRARY=/usr/lib64/atlas/libtatlas.so \
        -DAtlas_LAPACK_LIBRARY=/usr/lib64/atlas/libtatlas.so \
        -DHDF5_CXX_COMPILER_EXECUTABLE=/usr/bin/h5c++ \
        -DHDF5_C_COMPILER_EXECUTABLE=/usr/bin/h5cc \
        -DHDF5_DIFF_EXECUTABLE=/usr/bin/h5diff \
        -DOpenBabel3_INSTALL_PREFIX=/usr/local && \
    make -j${N_PROC} && \
    make -j${N_PROC} install

# A label for this package
LABEL tool.gnina=${gnina_version}

WORKDIR /root
