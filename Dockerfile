FROM python:3.7-slim-buster
ENV PYTHONUNBUFFERED 1

LABEL maintainer="Timo Halbesma <halbesma@MPA-Garching.MPG.DE>"

# Install system packages
WORKDIR /tidalshocks
RUN set -ex \
    && apt-get update \
\
    # Install Build/Runtime dependencies ...
    && apt-get install -y --no-install-recommends \
\
        # AMUSE dependencies (OpenMPI)
        build-essential gfortran python-dev \
        libopenmpi-dev openmpi-bin \
        libgsl-dev cmake libfftw3-3 libfftw3-dev \
        libgmp3-dev libmpfr6 libmpfr-dev \
        libhdf5-serial-dev hdf5-tools \
        git \
        # ... for a proper editor: vim, that is (runtime)
        vim emacs nano \
        # ... for the healthcheck (runtime)
        curl \
        # ... for monitoring (runtime)
        htop \
        # ... for video generation (runtime)
        ffmpeg \
        # ... lapack and blas for various packages (runtime)
        liblapack3 liblapack-dev libblas3 libblas-dev \
\
    # Create tidalshocks user to run uWSGI as non-root
    && groupadd -g 1000 tidalshocks \
    && useradd -r -u 1000 -g tidalshocks tidalshocks -s /bin/bash -d /tidalshocks

# Install python packages
COPY requirements.txt /tidalshocks/requirements.txt
RUN set -ex && \
    pip install --upgrade pip \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tidalshocks/requirements.txt

COPY . /tidalshocks
RUN chown -R tidalshocks:tidalshocks /tidalshocks

USER tidalshocks

ENTRYPOINT ["/tidalshocks/entrypoint.sh"]
