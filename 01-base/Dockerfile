FROM ubuntu:16.04

LABEL maintainer="achristie@informaticsmatters.com"

# The number of processors
# (for faster, multi-threaded make)
ARG n_proc=4
ENV N_PROC ${n_proc}

# Development Tools -----------------------------------------------------------

RUN apt-get -y update && \
    apt-get -y dist-upgrade && \
    apt-get -y install \
        build-essential \
        wget

# cmake (3.16.1) ---------------------------------------------------------------
# Essential for OpenBabel and libmolgrid builds
# libmolgrid requires 3.12 or higher.

RUN apt-get -y install \
        apt-transport-https \
        ca-certificates \
        gnupg \
        software-properties-common

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ xenial main' && \
    apt-get update && \
    apt-get -y install cmake

# Get rid of Python 3.5 (and 2.7) installed by the above.

RUN apt-get -y --purge remove python3.5
RUN apt-get -y --purge remove python2.7
RUN apt-get -y --purge remove \
        python3-minimal \
        python3.5-minimal
RUN apt-get -y --purge remove \
        python-minimal \
        python2.7-minimal

LABEL tool.cmake="v3.16.1"

# Python 3.5 ------------------------------------------------------------------

RUN apt-get -y install \
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
#
RUN apt-get -y install \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev
#
#WORKDIR /root/python3.7
#RUN wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz 2> /dev/null && \
#    tar xzf Python-3.7.5.tgz && \
#    cd Python-3.7.5 && \
#    ./configure --enable-optimizations --enable-shared && \
#    make -j${N_PROC} install && \
#    cd .. && \
#    rm -rf *
#
## The above results in python and pip at: -
##   -  /usr/local/bin/python3.7
##   -  /usr/local/bin/pip3.7
#RUN ln -s /usr/local/bin/python3.7 /usr/bin/python3 && \
#    ln -s /usr/local/bin/python3.7 /usr/bin/python && \
#    ln -s /usr/local/bin/pip3.7 /usr/bin/pip3 && \
#    ln -s /usr/local/bin/pip3.7 /usr/bin/pip
#
## Ensure the library path is setup
#ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/lib:/usr/lib:/usr/local/lib
#
#ENV PYTHON_INCLUDES /usr/local/include/python3.7m
#ENV CPLUS_INCLUDE_PATH ${CPLUS_INCLUDE_PATH}:${PYTHON_INCLUDES}
#ENV C_INCLUDE_PATH ${CPLUS_INCLUDE_PATH}:${PYTHON_INCLUDES}
#
#RUN pip install --upgrade pip
#
## A label for this package
#LABEL tool.python="v3.7.5"
#
# Wrap-up ---------------------------------------------------------------------

WORKDIR /root
