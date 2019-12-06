ARG base_image="informaticsmatters-deep-base-centos7:latest"
FROM ${base_image}

# Some modules other frameworks depend on...
RUN pip3 install \
        numpy==1.17.4 \
        pytest==5.3.1 \
        pyquaternion==0.9.5

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
LABEL tool.boost="v1.67.0"

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
LABEL tool.openbabel="v3.0.0"

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
LABEL tool.rdkit="v2019.09.1"

# Wrap-up ---------------------------------------------------------------------

WORKDIR /root
