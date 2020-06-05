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
        # ... for Arepo
        build-essential gfortran python-dev \
        mpich libmpich-dev \
        libgsl-dev cmake libfftw3-3 libfftw3-dev libfftw3-mpi-dev \
        libgmp3-dev libmpfr6 libmpfr-dev \
        libhdf5-serial-dev hdf5-tools \
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
    && pip install galpy==1.5.0


COPY . /tidalshocks
RUN chown -R tidalshocks:tidalshocks /tidalshocks

ENV SYSTYPE=Docker
ENV HYDRA_HOST_FILE=/tidalshocks/hostfile
ENV OMPI_MCA_rmaps_base_oversubscribe=1
USER tidalshocks

ENTRYPOINT ["/tidalshocks/entrypoint.sh"]
