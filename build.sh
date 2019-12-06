#!/usr/bin/env bash
#
# Build the chain of Docker images.
#
# Alan Christie
# December 2019

# If IMAGE_TAG is defined we'll use that as the tag,
# if not the images are built as "latest:
set -euxo pipefail

TAG=${IMAGE_TAG:-latest}

( cd base ; docker build . -t "informaticsmatters-deep-base-centos7:${TAG}" )
( cd rdkit ; docker build . -t "informaticsmatters-deep-rdkit-centos7:${TAG}" --build-arg "from_tag=${TAG}" )
( cd cuda ; docker build . -t "informaticsmatters-deep-cuda-centos7:${TAG}" --build-arg "from_tag=${TAG}" )
( cd gnina ; docker build . -t "informaticsmatters-deep-gnina-centos7:${TAG}" --build-arg "from_tag=${TAG}" )
