ARG CUDA_VERSION=11.4.3
ARG PYTHON_VERSION=3.8

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu20.04 as base

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt upgrade -y && \
    apt install -y build-essential && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install --no-install-recommends -y python3-pip libpq-dev python3-dev && \
    apt-get install --no-install-recommends -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev libgtk2.0-dev libgl1-mesa-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install -U setuptools

ENV PATH=$PATH:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib:/usr/local/cuda/lib64

COPY ./base /experiment

RUN ln -s /experiment/run.sh /usr/bin/run

RUN chmod +x /experiment/run.sh
RUN chown -R 42420:42420 /experiment