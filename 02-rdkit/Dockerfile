ARG from_image="informaticsmatters/deep-base-ubuntu"
ARG from_tag="latest"
FROM ${from_image}:${from_tag}

# System packages -------------------------------------------------------------

RUN apt-get -y install \
        git \
        libbz2-dev \
        lzma \
        wget

# Some modules other frameworks depend on...
RUN pip install \
        numpy==1.17.4 \
        pytest==5.3.1 \
        pyquaternion==0.9.5

# Boost (1.67.0) --------------------------------------------------------------

WORKDIR /root/boost
RUN wget https://dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.tar.gz 2> /dev/null && \
    tar -xzf boost_1_67_0.tar.gz
WORKDIR boost_1_67_0
RUN echo "using python : 3.5 ;" > /root/user-config.jam && \
    ./bootstrap.sh \
        --with-python=/usr/bin/python \
        --with-python-version=3.5 \
        --with-python-root=/usr/lib/python3.5 && \
    ./b2 -j${N_PROC} install && \
    cd .. && \
    rm -rf *

# By default installs to '/usr/local'
ENV BOOST_ROOT /usr/local

# A label for this package
LABEL tool.boost="v1.67.0"

# RapidJSON (1.1.0) -----------------------------------------------------------

WORKDIR /root

RUN git clone https://github.com/Tencent/rapidjson.git
WORKDIR rapidjson
RUN git checkout tags/v1.1.0

# RapidJSON is a header-only C++ library.
# You just need the include/rapidjson folder in the system include path.
ENV CPLUS_INCLUDE_PATH ${CPLUS_INCLUDE_PATH}:/root/rapidjson/include

# A label for this package
LABEL tool.rapidjson="v1.1.0"

# OpenBabel (3.0.0) -----------------------------------------------------------

# xml2 and cairo are probably not completely necessary,
# they're here to avoid some cmake warnings but the lack of these
# frameworks is probably harmless.
#RUN apt-get -y install \
#        libxml2-dev \
#        libcairo-dev

WORKDIR /root
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
RUN cmake .. \
        -DPYTHON_EXECUTABLE=/usr/bin/python \
        -DWITH_MAEPARSER=off && \
    make -j${N_PROC} && \
    make -j${N_PROC} install && \
    cd .. && \
    rm -rf *

# Update the linker library path
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/local/lib
# And include path
ENV CPLUS_INCLUDE_PATH ${CPLUS_INCLUDE_PATH}:/usr/local/include/openbabel3

# A label for this package
LABEL tool.openbabel="v3.0.0"

# RDKit (2019.09.1) -----------------------------------------------------------

# Jack uses 2018_03_04 but I get build problems with this
# relating to python and boost. At least 2019_09_01 builds.
WORKDIR /root/rdkit
ENV RDBASE /root/rdkit/rdkit-Release_2019_09_1
RUN wget https://github.com/rdkit/rdkit/archive/Release_2019_09_1.tar.gz 2> /dev/null && \
    tar -xzf Release_2019_09_1.tar.gz && \
    rm Release_2019_09_1.tar.gz

# Build...
WORKDIR ${RDBASE}/build
RUN cmake .. \
        -DCMAKE_CXX_FLAGS=-w \
        -DPYTHON_EXECUTABLE=/usr/bin/python \
        -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m \
        -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so \
        -DPYTHON_NUMPY_INCLUDE_PATH=/usr/local/lib/python3.5/dist-packages/numpy/core/include/ && \
    make -j${N_PROC} && \
    make -j${N_PROC} install && \
    cd .. && \
    find . -name "*.o" -exec rm -rf {} \;

# Update the linker library path
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${RDBASE}/lib
ENV CPLUS_INCLUDE_PATH ${CPLUS_INCLUDE_PATH}:${RDBASE}/Code

# A label for this package
LABEL tool.rdkit="v2019.09.1"

# Wrap-up ---------------------------------------------------------------------

WORKDIR /root
