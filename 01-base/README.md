# A convenient base image
A base container image for the deep/GPU workloads.

An Ubuntu 16.04 image that contains the following: -

-   gcc
-   cmake
-   Python 3.5

## Building

    $ docker build . -t informaticsmatters/deep-base-ubuntu-1604:latest --network=host 

## Running

    $ docker run --rm -it informaticsmatters/deep-rdkit-ubuntu-1604:latest bash
    
---
