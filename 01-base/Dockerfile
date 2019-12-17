FROM nvidia/cuda:10.2-devel-ubuntu16.04

LABEL maintainer="achristie@informaticsmatters.com"

# The number of processors
# (for faster, multi-threaded make)
ARG n_proc=4
ENV N_PROC ${n_proc}

ENV CUDA_ROOT_DIR /usr/local/cuda-10.2
ENV CUDA_BIN_DIR ${CUDA_ROOT_DIR}/bin
ENV CUDACXX ${CUDA_BIN_DIR}/nvcc

# cmake (3.16.1) --------------------------------------------------------------
# Essential for OpenBabel and libmolgrid builds
# libmolgrid requires 3.12 or higher.

RUN apt-get update && \
    apt-get -y --no-install-recommends install \
        apt-transport-https \
        ca-certificates \
        gnupg \
        software-properties-common \
        wget

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ xenial main' && \
    apt-get update && \
    apt-get -y install cmake

# Get rid of Python 2.7 installed by the above.

RUN apt-get -y --purge remove python2.7
RUN apt-get -y --purge remove \
        python-minimal \
        python2.7-minimal

LABEL tool.cmake="v3.16.1"

# Python 3.5 ------------------------------------------------------------------

RUN apt-get -y --no-install-recommends install \
        libpython3.5-dev \
        libpython3.5 \
        python3.5 \
        python3-pip

RUN ln -s /usr/bin/python3.5 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Ensure the library path is setup
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/lib:/usr/lib:/usr/local/lib

ENV PYTHON_INCLUDES /usr/include/python3.5m
ENV CPLUS_INCLUDE_PATH ${CPLUS_INCLUDE_PATH}:${PYTHON_INCLUDES}
ENV C_INCLUDE_PATH ${CPLUS_INCLUDE_PATH}:${PYTHON_INCLUDES}

RUN pip install --upgrade pip

# A label for this package
LABEL tool.python="v3.5.2"

# Python (3.7.5) --------------------------------------------------------------

RUN apt-get -y --no-install-recommendsinstall \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev

# Wrap-up ---------------------------------------------------------------------

WORKDIR /root
