#!/usr/bin/env bash

# Build the chain of Docker images.
#
# Alan Christie
# December 2019

set -euxo pipefail

# If IMAGE_TAG is defined we'll use that as the tag,
# if not the images are built as "latest"
TAG=${IMAGE_TAG:-latest}
# Get the number of physcal cores on the system
# (not execution threads, physical cores)
os_name=$(uname -s)
if [[ "${os_name}" == "Linux" ]]; then
  sockets=$(lscpu | grep Socket | tr -s ' ' | cut -f2 -d' ')
  per_socket_cores=$(lscpu | grep "per socket" | tr -s ' ' | cut -f4 -d' ')
  (( PROCESSORS = "${sockets}" * "${per_socket_cores}" ))
elif [[ "${os_name}" == "Darwin" ]]; then
  PROCESSORS="$(sysctl -n hw.physicalcpu)"
fi

( cd 01-base ; docker build . -t "informaticsmatters/deep-base-ubuntu-1604:${TAG}" --network=host --build-arg "n_proc=${PROCESSORS}")
( cd 02-boost ; docker build . -t "informaticsmatters/deep-boost-ubuntu-1604:${TAG}" --network=host --build-arg "from_tag=${TAG}" )
( cd 04-gnina ; docker build . -t "informaticsmatters/deep-gnina-ubuntu-1604:${TAG}" --network=host --build-arg "from_tag=${TAG}" )
( cd 05-app ; docker build . -t "informaticsmatters/deep-app-ubuntu-1604:${TAG}" --network=host --build-arg "from_tag=${TAG}" )
