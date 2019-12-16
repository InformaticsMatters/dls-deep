ARG from_image="informaticsmatters/deep-gnina-ubuntu"
ARG from_tag="latest"
FROM ${from_image}:${from_tag}

# App -------------------------------------------------------------------------

WORKDIR /root/train
ADD files.tar.gz ./
WORKDIR fragalysis_test_files
COPY requirements.txt ./
RUN pip3 install -r requirements.txt && \
    chmod -R 0755 *.py && \
    chmod -R 0755 *.sh

ENV PYTHONPATH /usr/local/python:${PYTHONPATH}
