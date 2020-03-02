ARG from_image="informaticsmatters/deep-gnina-ubuntu-1604"
ARG from_tag="latest"
FROM ${from_image}:${from_tag}

# App -------------------------------------------------------------------------

WORKDIR /train
ADD files.tar.gz ./

RUN chmod 755 /train &&\
  chmod 775 -R /train/fragalysis_test_files &&\
  chmod 777 /train/fragalysis_test_files &&\
  chown -R 0.0 /train

WORKDIR fragalysis_test_files
COPY requirements.txt ./
RUN pip install -r requirements.txt && \
    chmod -R 0755 *.py && \
    chmod -R 0755 *.sh

# Wrap-up ---------------------------------------------------------------------

# At this point the user can simply enter the container
# (where they end up in the 'fragalysis_test_files' directory)
# and run...
#
#   ./call_main.sh

CMD "./call_main.sh"
