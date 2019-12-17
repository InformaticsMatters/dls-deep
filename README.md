# dls-deep
Images for gnina/caffe GPU workloads using the [NVIDIA Docker] runtime.

Consisting of a series of base image (that might be generally useful
independently) and a series of additional images that add further GPU
and deep-learning (artificial neural-net) frameworks.

## Building
Build the chain of images with: -

    $ ./build.sh
    
>   Building will take a few minutes on a 15-core machine,
    and significantly longer on smaller machines. For building, the scripts
    automatically use all the cores available on the host.

If you want to tag the images with something other than `latest`: -

    $ IMAGE_TAG=2019.12 ./build.sh

## Execution
To run the container on a GPU host using that has the nvidia docker driver: -

    $ docker run --gpus all --rm -it informaticsmatters/deep-app-ubuntu-1604:latest bash
    # ./call_main.sh
    [...]
    Finished! Exiting...
    
---

[nvidia docker]: https://github.com/NVIDIA/nvidia-docker
[nvidia runtime]: https://github.com/NVIDIA/nvidia-docker/wiki/Usage
[cuda and docker]: https://medium.com/@adityathiruvengadam/cuda-docker-%EF%B8%8F-for-deep-learning-cab7c2be67f9
[using nvidia containers]: https://marmelab.com/blog/2018/03/21/using-nvidia-gpu-within-docker-container.html
