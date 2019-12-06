FROM centos:centos7.7.1908

LABEL maintainer="achristie@informaticsmatters.com"

# The number of processors
# (for faster, multi-threaded make)
ARG N_PROC=4

# General dependencies --------------------------------------------------------
# Please keep this list (roughly) alphabetical
# so it's easy to check what we're installing
# when we want to add more.

RUN yum -y -q install \
        wget \

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

LABEL tool.gnu="v8.3.1"

# cmake (3.15.5) --------------------------------------------------------------
# Essential for OpenBabel and libmolgrid builds

WORKDIR /root/cmake
RUN source scl_source enable devtoolset-8 && \
    wget https://cmake.org/files/v3.15/cmake-3.15.5.tar.gz 2> /dev/null && \
    tar -xf cmake-3.* && \
    rm cmake-3.15.5.tar.gz && \
    cd cmake-3.* && \
    ./bootstrap --prefix=/usr/local && \
    make -j${N_PROC} && \
    make -j${N_PROC} install && \
    rm -rf *

LABEL tool.cmake="v3.15.5"

# Python (3.7.5) --------------------------------------------------------------

RUN yum -y -q install \
        bzip2-devel \
        libffi-devel \
        openssl-devel

WORKDIR /root/python3.7
RUN wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz 2> /dev/null && \
    tar xzf Python-3.7.5.tgz && \
    cd Python-3.7.5 && \
    source scl_source enable devtoolset-8 && \
    ./configure --enable-optimizations --enable-shared && \
    make -j${N_PROC} altinstall && \
    cd .. && \
    rm -rf *

# The above results in python and pip at: -
#   -  /usr/local/bin/python3.7
#   -  /usr/local/bin/pip3.7
RUN ln -s /usr/local/bin/python3.7 /usr/bin/python3 && \
    ln -s /usr/local/bin/pip3.7 /usr/bin/pip3

# Ensure the library path is setup
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/lib:/usr/lib:/usr/local/lib

ENV PYTHON_INCLUDES /usr/local/include/python3.7m
ENV CPLUS_INCLUDE_PATH ${CPLUS_INCLUDE_PATH}:${PYTHON_INCLUDES}
ENV C_INCLUDE_PATH ${CPLUS_INCLUDE_PATH}:${PYTHON_INCLUDES}

# A label for this package
LABEL tool.python="v3.7.5"

# Wrap-up ---------------------------------------------------------------------

ENV N_PROC ${N_PROC}

WORKDIR /root
