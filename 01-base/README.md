# A convenient base image
A base container image for the deep/GPU workloads.

A CentOS 8 image that contains the following: -

-   gcc
-   cmake
-   Python 3.7

## Building

    $ docker build . -t informaticsmatters/deep-base-centos8:latest --network=host
    
---
