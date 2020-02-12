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

## Building your own variant
This repository's directories result in a functional GPU-enabled container
image from information and help provided by Jack Scantlebury and
Diamond Light Systems.

Significant effort was required to translate the working research code
into a runnable container and very little time spent minimising or optimising the
built - that wasn't the objective. The objective was to run the GPU application
from within a container image - the image size and efficient of construction
was unimportant.

Nevertheless, for further work the following (estimated) dependencies
may be useful for those wishing to create a compact image or compile
the solution to a different medium.

### Build-time dependencies
The individual Dockerfiles will document all the tools required to build
(and run) the image.
  
### Run-time dependencies
This is a little more tricky as a full chain of run-time dependencies is
not evident from the Dockerfile content. The package manager and Python
may pull in many packages and modules based on the user's requirements.
Nevertheless a _best guess_ at a set of _user level_ dependencies follows.
 
A word of **caution** - this list is _guide_. A full list of dependencies will
probably require closer examination and/or the author's input. Nevertheless
this is a reasonable starting point for a list of key run-time dependencies: -

-   `CUDA 10.2`
-   `Python 3.5`. The dls-deep image is built with 3.5 although I suspect it
    might work with later version like 3.8 but that is untested.
-   Python Modules
    - `numpy == 1.17.4`
    - `pyquaternion == 0.9.5`
-   `boost 1.67.0`
-   `openbabel 3.0.0`
-   `libmolgrid` . There were no formal releases at the time we concluded our
    main phase of development. We currently use a fixed commit `a5bd251`.
    We cannot confirm whether the new (official) versions of `v0.1` or
    `v0.1.1` will work.
-   `gnina`. There are no official releases of **gnina**. We
    used the fixed commit `16ce46d`.
    
It's unclear as to whether the following are run-time or build-time
dependencies, but are listed here, _just in case_.

-   `rapidjson 1.1.0`

>   The build includes a large number of system packages obtained with the
    package manager (`apt-get`). It's extremely difficult to know whether
    they're for running, building or both. If all else fails add all the
    packages identified by the `apt-get` lines in the various Dockerfiles.
    
Be prepared to be forced to compile some items from source as several are very
much _bleeding edge_ (some without any formal release) and, as such, are
available only in source form, not as packages. For this you will need,
at the very least: -

-   `gcc (v8.3.1 or better)`
-   `cmake (at least v3.15.5 for libmolgrid)`

---

[nvidia docker]: https://github.com/NVIDIA/nvidia-docker
[nvidia runtime]: https://github.com/NVIDIA/nvidia-docker/wiki/Usage
[cuda and docker]: https://medium.com/@adityathiruvengadam/cuda-docker-%EF%B8%8F-for-deep-learning-cab7c2be67f9
[using nvidia containers]: https://marmelab.com/blog/2018/03/21/using-nvidia-gpu-within-docker-container.html
