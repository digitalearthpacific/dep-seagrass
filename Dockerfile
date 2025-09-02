FROM ghcr.io/osgeo/gdal:ubuntu-full-3.10.0

# Don't use old pygeos
ENV USE_PYGEOS=0

RUN apt-get update && apt-get install -y \
    htop \
    python3-pip \
    python3-dev \
    git \
    libpq-dev \
    ca-certificates \
    build-essential \
    && apt-get autoclean \
    && apt-get autoremove \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

ADD requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache --break-system-packages -r /tmp/requirements.txt

ADD . /code

WORKDIR /code
