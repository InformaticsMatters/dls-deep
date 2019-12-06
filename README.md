# dls-deep
Images for gnina/caffe GPU workloads.

Consisting of a base image (that might be generally useful) and a series
of additional images that add further GPU and deep-learning
(artificial neural-net) frameworks.

## Building
Build the chain of images with: -

    $ ./build.sh
    
If you want to tag the images with something other than `latest`: -

    $ IMAGE_TAG=2019.12 ./build.sh

---

