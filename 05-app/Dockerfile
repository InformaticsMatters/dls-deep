ARG from_image="informaticsmatters/deep-gnina-ubuntu-1604"
ARG from_tag="latest"
FROM ${from_image}:${from_tag}

# App -------------------------------------------------------------------------

WORKDIR /root/train
ADD files.tar.gz ./
WORKDIR fragalysis_test_files
COPY requirements.txt ./
RUN pip install -r requirements.txt && \
    chmod -R 0755 *.py && \
    chmod -R 0755 *.sh

ENV PYTHONPATH /usr/local/python:${PYTHONPATH}

# Wrap-up ---------------------------------------------------------------------

# At this point the user can simply enter the container
# (where they end up in the 'fragalysis_test_files' directory)
# and run...
#
#   ./call_main.sh
