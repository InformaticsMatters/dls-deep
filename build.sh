#!/usr/bin/env bash

# Build the chain of Docker images.
#
# Alan Christie
# December 2019

set -euxo pipefail

# If IMAGE_TAG is defined we'll use that as the tag,
# if not the images are built as "latest"
TAG=${IMAGE_TAG:-latest}
# Also allow the number of processors to be defined
# Default is 4
PROCESSORS=${PROC:-4}

( cd base ; docker build . -t "informaticsmatters/deep-base-centos8:${TAG}" --build-arg "n_proc=${PROCESSORS}")
( cd rdkit ; docker build . -t "informaticsmatters/deep-rdkit-centos8:${TAG}" --build-arg "from_tag=${TAG}" )
( cd cuda ; docker build . -t "informaticsmatters/deep-cuda-centos8:${TAG}" --build-arg "from_tag=${TAG}" )
( cd gnina ; docker build . -t "informaticsmatters/deep-gnina-centos8:${TAG}" --build-arg "from_tag=${TAG}" )
( cd app ; docker build . -t "informaticsmatters/deep-app-centos8:${TAG}" --build-arg "from_tag=${TAG}" )
