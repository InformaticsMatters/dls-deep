#!/usr/bin/env bash

# Setup the physical host for execution.
# A script to be execute as the root user: -
#
#   $ sudo -i
#   $ ./setup-physical.sh
#
# Alan Christie
# December 2019

set -euxo pipefail

# This assumes the host has the CUDA toolkit installed
# so the Dockerfile CUDA image actions are not repeated here.
#
# To verify this we must have nvcc...

export CUDA_ROOT_DIR=/usr/local/cuda-10.2
export CUDA_BIN_DIR=${CUDA_ROOT_DIR}/bin
export CUDACXX=${CUDA_BIN_DIR}/nvcc

if [ ! -d "${CUDA_ROOT_DIR}" ]; then
    echo "Missing ${CUDA_ROOT_DIR}..."
    fail
fi

${CUDACXX} --version > /dev/null

# ...and we must be abel to run the built-in example 'deviceQuery'...
${CUDA_ROOT_DIR}/extras/demo_suite/deviceQuery

# Get the number of physcal cores on the system
# (not execution threads, physical cores)
sockets=$(lscpu | grep Socket | tr -s ' ' | cut -f2 -d' ')
per_socket_cores=$(lscpu | grep "per socket" | tr -s ' ' | cut -f4 -d' ')
(( N_PROC = "${sockets}" * "${per_socket_cores}" ))

mkdir -p setup
pushd setup || exit

###############################################################################
# BASE IMAGE ACTIONS
###############################################################################

# cmake (3.16.1) --------------------------------------------------------------

apt-get -y install \
    apt-transport-https \
    ca-certificates \
    gnupg \
    software-properties-common

wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
apt-add-repository 'deb https://apt.kitware.com/ubuntu/ xenial main'
apt-get update
apt-get -y install cmake

# Get rid of Python 2.7 installed by the above.

apt-get -y --purge remove python2.7
apt-get -y --purge remove \
    python-minimal \
    python2.7-minimal

# check...
cmake --version > /dev/null

# Python (3.5.2) --------------------------------------------------------------

apt-get -y install \
    libpython3.5-dev \
    libpython3.5 \
    python3.5 \
    python3-pip

python3_path=$(command -v python3.5)
pip3_path=$(command -v pip3)
ln -sf "${python3_path}" /usr/bin/python
ln -sf "${pip3_path}" /usr/bin/pip

# Ensure the library paths are setup

LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-''}
CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH:-''}
C_INCLUDE_PATH=${C_INCLUDE_PATH:-''}

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/lib:/usr/lib:/usr/local/lib

export PYTHON_INCLUDES=/usr/include/python3.5m
export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:${PYTHON_INCLUDES}
export C_INCLUDE_PATH=${C_INCLUDE_PATH}:${PYTHON_INCLUDES}

pip install --upgrade pip

# Extra stuff (for subsequent builds)

apt-get -y install \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev

###############################################################################
# RDKIT IMAGE ACTIONS
###############################################################################

apt-get -y install \
    git \
    libbz2-dev \
    lzma \
    wget

# Some modules other frameworks depend on...
pip install \
    numpy==1.17.4 \
    pytest==5.3.1 \
    pyquaternion==0.9.5

# Boost (1.67.0) --------------------------------------------------------------

mkdir -p boost
pushd boost || exit
wget https://dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.tar.gz 2> /dev/null
tar -xzf boost_1_67_0.tar.gz
cd boost_1_67_0 || exit
echo "using python : 3.5 ;" > ~/user-config.jam
./bootstrap.sh \
    --with-python=/usr/bin/python \
    --with-python-version=3.5 \
    --with-python-root=/usr/lib/python3.5
./b2 -j${N_PROC} install

# By default installs to '/usr/local'
export BOOST_ROOT=/usr/local

popd || exit

# RapidJSON (1.1.0) -----------------------------------------------------------

rm -rf rapidjson
git clone https://github.com/Tencent/rapidjson.git
pushd rapidjson || exit
git checkout tags/v1.1.0

# RapidJSON is a header-only C++ library.
# You just need the include/rapidjson folder in the system include path.
export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:${PWD}/include

popd || exit

# OpenBabel (3.0.0) -----------------------------------------------------------

rm -rf openbabel
git clone https://github.com/openbabel/openbabel.git
pushd openbabel || exit
git checkout tags/openbabel-3-0-0

mkdir -p build
cd build || exit
cmake .. \
    -DPYTHON_EXECUTABLE=/usr/bin/python \
    -DWITH_MAEPARSER=off
make -j${N_PROC}
make -j${N_PROC} install

# Update the linker library path
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib
# And include path
export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:/usr/local/include/openbabel3

popd || exit

###############################################################################
# GNINA IMAGE ACTIONS
###############################################################################

# libmolgrid (latest) ---------------------------------------------------------

libmolgrid_version="a5bd251"

rm -rf libmolgrid
git clone https://github.com/gnina/libmolgrid.git

# Nobble test builds and testing because these appear not to build or run.
# By commenting-out the 'enable_testing' and 'add_subdirectory(test)' lines...
pushd libmolgrid || exit
git checkout ${libmolgrid_version}
sed -i 's/enable_testing/#enable_testing/g' CMakeLists.txt
sed -ri 's/add_subdirectory[(]test[)]/#add_subdirectory(test)/g' CMakeLists.txt

mkdir -p build
cd build || exit
cmake .. \
    -DCMAKE_CUDA_COMPILER=${CUDACXX} \
    -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_ROOT_DIR} \
    -DPYTHON_EXECUTABLE=/usr/bin/python \
    -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m \
    -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so
make -j${N_PROC}
make -j${N_PROC} install

popd || exit

# gnina (latest) --------------------------------------------------------------

gnina_version="16ce46d"

apt-get -y install \
    libeigen3-dev \
    libgoogle-glog-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libhdf5-serial-dev \
    libatlas-base-dev

rm -rf gnina
git clone https://github.com/gnina/gnina.git

# Nobble test builds and testing because these appear not to build or run.
# By commenting-out the 'enable_testing' and 'add_subdirectory(test)' lines...
pushd gnina || exit
git checkout ${gnina_version}
sed -i 's/enable_testing/#enable_testing/g' CMakeLists.txt
sed -ri 's/add_subdirectory[(]test[)]/#add_subdirectory(test)/g' CMakeLists.txt

mkdir -p build
cd build || exit
cmake .. \
    -DCMAKE_CUDA_COMPILER=${CUDACXX} \
    -DPYTHON_EXECUTABLE=/usr/bin/python \
    -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m \
    -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so \
    -DOpenBabel3_INSTALL_PREFIX=/usr/local
make -j${N_PROC}
make -j${N_PROC} install
make -j${N_PROC} pycaffe

popd || exit

###############################################################################
# APP IMAGE ACTIONS
###############################################################################

popd || exit

pushd 05-app || exit
pip install -r requirements.txt
tar -xvf files.tar.gz
popd || exit

PYTHONPATH=${PYTHONPATH:-''}
export PYTHONPATH=/usr/local/python:${PYTHONPATH}

###############################################################################
# That's All Folks!
###############################################################################

# Finally, we must still be able to run the built-in example 'deviceQuery'...
${CUDA_ROOT_DIR}/extras/demo_suite/deviceQuery
