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
        # ... a compiler (build)
        build-essential gcc \
        # ... for a proper editor: vim, that is (runtime)
        vim emacs nano \
        # ... for the healthcheck (runtime)
        curl \
        # ... for monitoring (runtime)
        htop \
        # ... for video generation (runtime)
        ffmpeg \
        # ... for version control (runtime)
        git \
        # ... lapack and blas for various packages (runtime)
        liblapack3 liblapack-dev libblas3 libblas-dev \
        # ... gsl for various packages (runtime)
        libgsl-dev \
\
    # Create tidalshocks user to run uWSGI as non-root
    && groupadd -g 1000 tidalshocks \
    && useradd -r -u 1000 -g tidalshocks tidalshocks -s /bin/bash -d /tidalshocks

# Install python packages for Django
COPY requirements.txt /tidalshocks/requirements.txt
RUN set -ex && \
    pip install --upgrade pip \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tidalshocks/requirements.txt

COPY . /tidalshocks
RUN chown -R tidalshocks:tidalshocks /tidalshocks

USER tidalshocks

ENTRYPOINT ["/tidalshocks/entrypoint.sh"]
