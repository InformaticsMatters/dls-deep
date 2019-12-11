ARG from_image="informaticsmatters-deep-gnina-centos7"
ARG from_tag="latest"
FROM ${from_image}:${from_tag}

# App -------------------------------------------------------------------------

WORKDIR /root/train
COPY files ./
RUN pip3 install -r requirements.txt && \
    chmod -R 0755 *.py && \
    chmod -R 0755 *.sh

ENV PYTHONPATH /usr/local/python/caffe:/usr/local/lib/python3.7/site-packages

# Wrap-up ---------------------------------------------------------------------

WORKDIR /root/train
