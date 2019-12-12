FROM centos:8

LABEL maintainer="achristie@informaticsmatters.com"

# The number of processors
# (for faster, multi-threaded make)
ARG n_proc=4
ENV N_PROC ${n_proc}

# Dveleopment Tools -----------------------------------------------------------

RUN dnf -y group install "Development Tools" && \
    dnf -y --nogpg install \
        wget \
        which

# cmake (3.15.5) --------------------------------------------------------------
# Essential for OpenBabel and libmolgrid builds

WORKDIR /root/cmake
RUN wget https://cmake.org/files/v3.15/cmake-3.15.5.tar.gz 2> /dev/null && \
    tar -xf cmake-3.* && \
    rm cmake-3.15.5.tar.gz && \
    cd cmake-3.* && \
    ./bootstrap --prefix=/usr/local && \
    make -j${N_PROC} && \
    make -j${N_PROC} install && \
    rm -rf *

LABEL tool.cmake="v3.15.5"

# Python (3.7.5) --------------------------------------------------------------

RUN dnf -y --nogpg install \
        bzip2-devel \
        libffi-devel \
        openssl-devel

WORKDIR /root/python3.7
RUN wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz 2> /dev/null && \
    tar xzf Python-3.7.5.tgz && \
    cd Python-3.7.5 && \
    ./configure --enable-optimizations --enable-shared && \
    make -j${N_PROC} install && \
    cd .. && \
    rm -rf *

# The above results in python and pip at: -
#   -  /usr/local/bin/python3.7
#   -  /usr/local/bin/pip3.7
RUN ln -s /usr/local/bin/python3.7 /usr/bin/python3 && \
    ln -s /usr/local/bin/python3.7 /usr/bin/python && \
    ln -s /usr/local/bin/pip3.7 /usr/bin/pip3 && \
    ln -s /usr/local/bin/pip3.7 /usr/bin/pip

# Ensure the library path is setup
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/lib:/usr/lib:/usr/local/lib

ENV PYTHON_INCLUDES /usr/local/include/python3.7m
ENV CPLUS_INCLUDE_PATH ${CPLUS_INCLUDE_PATH}:${PYTHON_INCLUDES}
ENV C_INCLUDE_PATH ${CPLUS_INCLUDE_PATH}:${PYTHON_INCLUDES}

RUN pip install --upgrade pip

# A label for this package
LABEL tool.python="v3.7.5"

# Wrap-up ---------------------------------------------------------------------

WORKDIR /root
