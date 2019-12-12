# The final (application) image
A container image for the application, which sits on top of gnina.

Adds the following: -

-   Application scripts
-   Execution data

Execution is simply accomplished by running `./call_main.sh` from
the landing directory.

## Building

    $ docker build . -t informaticsmatters-deep-app-centos8:latest
    
---
