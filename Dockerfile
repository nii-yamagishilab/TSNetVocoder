From nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-dev=3.5.1-3 \
    python3-pip=8.1.1-2ubuntu0.4 && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install -U pip==10.0.1
RUN pip3 install setuptools==39.2.0
RUN pip3 install \
    numpy==1.14.5 \
    pathlib==1.0.1 \
    wave==0.0.2 \
    matplotlib==2.2.2 \
    chainer==4.1.0 \
    cupy==4.1.0
