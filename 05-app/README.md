# The final (application) image
A container image for the application, which sits on top of gnina.

Adds the following: -

-   Application scripts
-   Execution data

## Building

    $ docker build . -t informaticsmatters/deep-app-ubuntu:latest --network=host
    
---
