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

( cd 01-base ; docker build . -t "informaticsmatters/deep-base-ubuntu:${TAG}" --network=host --build-arg "n_proc=${PROCESSORS}")
( cd 02-rdkit ; docker build . -t "informaticsmatters/deep-rdkit-ubuntu:${TAG}" --network=host --build-arg "from_tag=${TAG}" )
( cd 03-cuda ; docker build . -t "informaticsmatters/deep-cuda-ubuntu:${TAG}" --network=host --build-arg "from_tag=${TAG}" )
( cd 04-gnina ; docker build . -t "informaticsmatters/deep-gnina-ubuntu:${TAG}" --network=host --build-arg "from_tag=${TAG}" )
( cd 05-app ; docker build . -t "informaticsmatters/deep-app-ubuntu:${TAG}" --network=host --build-arg "from_tag=${TAG}" )
