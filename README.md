# dls-deep
Images for gnina/caffe GPU workloads.

Consisting of a series of base image (that might be generally useful
independently) and a series of additional images that add further GPU
and deep-learning (artificial neural-net) frameworks.

## Building
Build the chain of images with: -

    $ ./build.sh
    
If you want to tag the images with something other than `latest` on a 30-core
machine: -

    $ IMAGE_TAG=2019.12 PROC=30 ./build.sh

>   It's a long build. If your're building from scratch and you should prepare
    for a build time of approximately 80 minutes using a decent 4-core machine. 
    
---
