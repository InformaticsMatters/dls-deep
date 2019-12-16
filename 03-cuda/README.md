# A convenient base image
A base container image for the deep/GPU workloads.

Adds the following to the image: -

-   The NVIDIA CUDA Toolkit

## Building

    $ docker build . -t informaticsmatters/deep-cuda-ubuntu:latest

## Running

    $ docker run --rm -it informaticsmatters/deep-cuda-ubuntu:latest bash
    
---
