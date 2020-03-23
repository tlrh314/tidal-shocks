FROM python:3.8-slim-buster
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
        # ... for AMUSE (OpenMPI)
        build-essential gfortran python-dev \
        libopenmpi-dev openmpi-bin \
        libgsl-dev cmake libfftw3-3 libfftw3-dev libfftw3-mpi-dev \
        libgmp3-dev libmpfr6 libmpfr-dev \
        libhdf5-serial-dev libhdf5-openmpi-dev hdf5-tools \
        git \
        # ... backend for matplotlib.pyplot
        tk \ 
        # ... for a proper editor: vim, that is (runtime)
        vim emacs nano \
        # ... for the healthcheck (runtime)
        curl \
        # ... for monitoring (runtime)
        htop \
        # ... for video generation (runtime)
        ffmpeg \
        # ... for LaTeX support, pyplot and jupyter download as pdf via latex (runtime)
        texlive texlive-latex-extra texlive-generic-extra dvipng pandoc texlive-xetex \
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
    && pip install --no-cache-dir --upgrade pip numpy \
    && pip install --no-cache-dir -r /tidalshocks/requirements.txt \
    && jupyter contrib nbextension install --system \
    && jupyter nbextension enable execute_time/ExecuteTime \
    && jupyter nbextension enable autosavetime/main \
    && jupyter nbextension enable ruler/main \
    && jupyter nbextension enable toc2/main \
    && jupyter nbextension enable livemdpreview/livemdpreview \
\
    && pip install \
    amuse-bhtree==13.1.0 \
    amuse-fi==13.1.0 \
    amuse-framework==13.1.0 \
    amuse-gadget2==13.1.0  \
    amuse-galactics==13.1.0  \
    amuse-hermite==13.1.0 \
    amuse-hop==13.1.0 \
    amuse-huayno==13.1.0 \
    amuse-mercury==13.1.0 \
    amuse-ph4==13.1.0 \
    amuse-phigrape==13.1.0 \
    amuse-smalln==13.1.0 \
    galpy==1.5.0 \
\
    && echo "localhost slots=100" >> /etc/openmpi/openmpi-default-hostfile


COPY . /tidalshocks
RUN chown -R tidalshocks:tidalshocks /tidalshocks

ENV SYSTYPE=Docker
USER tidalshocks

ENTRYPOINT ["/tidalshocks/entrypoint.sh"]
