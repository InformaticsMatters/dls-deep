# A gnina/caffe image
A container image for gnina (from source code) based on a
corresponding base image.

Adds the following to a CentOS image: -

-   Boost
-   OpenBabel
-   RDKit 

## Building

    $ docker build . -t informaticsmatters/deep-rdkit-centos8:latest --network=host
    
---
