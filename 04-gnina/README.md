# A gnina/caffe image
A container image for gnina (from source code) based on a
corresponding base image.

Adds the following to the image: -

-   libmolgrid
-   gnina

## Building

    $ docker build . -t informaticsmatters/deep-gnina-ubuntu:latest --network=host

## Running

    $ docker run --rm -it informaticsmatters/deep-gnina-ubuntu:latest bash
    
---
