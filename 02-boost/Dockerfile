ARG from_image="informaticsmatters/deep-base-ubuntu-1604"
ARG from_tag="latest"
FROM ${from_image}:${from_tag}

# System packages -------------------------------------------------------------

RUN apt-get -y --no-install-recommends install \
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
    tar -xzf boost_1_67_0.tar.gz && \
    rm boost_1_67_0.tar.gz
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

WORKDIR /root
RUN git clone https://github.com/openbabel/openbabel.git
WORKDIR openbabel
RUN git checkout tags/openbabel-3-0-0

WORKDIR build
RUN cmake .. \
        -DPYTHON_EXECUTABLE=/usr/bin/python \
	-DPYTHON_BINDINGS=ON \
        -DWITH_MAEPARSER=off && \
    make -j${N_PROC} && \
    make -j${N_PROC} install && \
    cd .. && \
    rm -rf *
ENV PYTHONPATH /usr/local/lib

# Update the linker library path
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/local/lib
# And include path
ENV CPLUS_INCLUDE_PATH ${CPLUS_INCLUDE_PATH}:/usr/local/include/openbabel3

# A label for this package
LABEL tool.openbabel="v3.0.0"

# Wrap-up ---------------------------------------------------------------------

WORKDIR /root
