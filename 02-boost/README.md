# A gnina/caffe image
A container image for gnina (from source code) based on a
corresponding base image.

Adds the following to the image: -

-   Boost
-   OpenBabel
-   RDKit 

## Building

    $ docker build . -t informaticsmatters/deep-boost-ubuntu-1604:latest --network=host 

## Running

    $ docker run --rm -it informaticsmatters/deep-boost-ubuntu-1604:latest bash
    
---
