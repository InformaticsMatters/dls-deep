# An image for gina applications based on the Informatic Matters
# base comtainer image, whcih includes v8 development tools (gcc etc.).
# The latest tools need to be 'sourced' for each RUN command where
# they're reuired.

ARG base_image="informaticsmatters-deep-base-centos7:latest"
FROM ${base_image}

ENV PYTHON_INCLUDES /usr/local/include/python3.7m
ENV CPLUS_INCLUDE_PATH ${CPLUS_INCLUDE_PATH}:${PYTHON_INCLUDES}
ENV C_INCLUDE_PATH ${CPLUS_INCLUDE_PATH}:${PYTHON_INCLUDES}

# Python (3.7.5) --------------------------------------------------------------

RUN yum -y -q install \
        bzip2-devel \
        libffi-devel \
        openssl-devel

WORKDIR /home/python3.7
RUN wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz 2> /dev/null && \
    tar xzf Python-3.7.5.tgz && \
    cd Python-3.7.5 && \
    source scl_source enable devtoolset-8 && \
    ./configure --enable-optimizations --enable-shared && \
    make -j${N_PROC} altinstall

# The above results in python and pip at: -
#   -  /usr/local/bin/python3.7
#   -  /usr/local/bin/pip3.7
RUN ln -s /usr/local/bin/python3.7 /usr/bin/python3 && \
    ln -s /usr/local/bin/pip3.7 /usr/bin/pip3

# Ensure the library path is setup
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/lib:/usr/lib:/usr/local/lib

# Some modules other framworks may depend on...
RUN pip3 install \
        numpy==1.17.4 \
        pytest==5.3.1 \
        pyquaternion==0.9.5

# A label for this package
LABEL python="v3.7.5"

# Boost (1.67.0) --------------------------------------------------------------

WORKDIR /home/boost
RUN wget https://dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.tar.gz 2> /dev/null && \
    tar -xzf boost_1_67_0.tar.gz
WORKDIR boost_1_67_0
RUN echo "using python : 3.7 ;" > /root/user-config.jam && \
    source scl_source enable devtoolset-8 && \
    ./bootstrap.sh \
        --with-python=/usr/bin/python3 \
        --with-python-version=3.7 \
        --with-python-root=/usr/local/lib/python3.7 && \
    ./b2 -j${N_PROC} install

# By default installs to '/usr/local'
ENV BOOST_ROOT /usr/local

# A label for this package
LABEL boost="v1.67.0"

# OpenBabel (3.0.0) -----------------------------------------------------------

# Use system rapidjson rather than the built-in one?
# libxml2 and cairo are probably not completely necessary,
# they're her eto avoid some cmake warnings but the lack of these
# frameworks is probably harmless.
RUN yum -y -q install \
        rapidjson \
        rapidjson-devel \
        libxml2-devel \
        cairo-devel

WORKDIR /home
RUN git clone https://github.com/openbabel/openbabel.git
WORKDIR openbabel
RUN git checkout tags/openbabel-3-0-0

# MAEPARSER!!!!!!!
# WHY HAS THIS STOPPED WORKIMG!!!!!!!!
# WITHOUT IT GNINA BUID FAILES
# IT GETS DOWNLOADED BUT DOESN'T GET DECOMPRESSED
#
# Manually install maeparser
# as this seems to be failign wnen building from source
# (cmake attempts to download the file but it ends up being zero-length
# and not unpacked)
#WORKDIR external
#RUN wget https://github.com/schrodinger/maeparser/archive/v1.1.tar.gz 2> /dev/null && \
#    tar -xzf v1.1.tar.gz && \
#    rm v1.1.tar.gz
#
# FOR NOW I'VE SET 'WITH_MAEPARSER=off'

WORKDIR build
RUN source scl_source enable devtoolset-8 && \
    cmake .. \
        -DPYTHON_EXECUTABLE=/usr/bin/python3 \
        -DWITH_MAEPARSER=off && \
    make -j${N_PROC} && \
    make -j${N_PROC} install

# Update the linker library path
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/local/lib
# And include path
ENV CPLUS_INCLUDE_PATH ${CPLUS_INCLUDE_PATH}:/usr/local/include/openbabel3

# A label for this package
LABEL openbabel="v3.0.0"

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

# RDkit (2019.09.1) -----------------------------------------------------------

# Jack uses 2018_03_04 but I get build problems with this
# relating to python and boost. At least 2019_09_01 builds.
WORKDIR /home/rdkit
ENV RDBASE /home/rdkit/rdkit-Release_2019_09_1
RUN wget https://github.com/rdkit/rdkit/archive/Release_2019_09_1.tar.gz 2> /dev/null && \
    tar -xzf Release_2019_09_1.tar.gz

# Build...
WORKDIR rdkit-Release_2019_09_1/build
RUN source scl_source enable devtoolset-8 && \
    cmake .. \
        -DCMAKE_CXX_FLAGS=-w \
        -DPYTHON_EXECUTABLE=/usr/bin/python3 \
        -DPYTHON_INCLUDE_DIR=/usr/local/include/python3.7m \
        -DPYTHON_LIBRARY=/usr/local/lib/libpython3.7m.so \
        -DPYTHON_NUMPY_INCLUDE_PATH=/usr/local/lib/python3.7/site-packages/numpy/core/include && \
    make -j${N_PROC} && \
    make -j${N_PROC} install

# On CentOS
# we need to make additional links to resemble the UBUNTU names...
WORKDIR /home/rdkit/rdkit-Release_2019_09_1/lib
RUN for i in $(ls -1 *.so.1.2019.09.1); \
    do \
        name=`basename $i .so.1.2019.09.1`; \
        namef=`echo $name | sed 's/RDKit//g'`; \
        ln -s $i ${namef}.so.1; \
        ln -s ${namef}.so.1 ${namef}.so; \
    done

# Update the linker library path
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${RDBASE}/lib
ENV CPLUS_INCLUDE_PATH ${CPLUS_INCLUDE_PATH}:${RDBASE}/Code

# A label for this package
LABEL rdkit="v2019.09.1"

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
LABEL gnina=${gnina_version}
