ARG CUDA_VERSION=11.7.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu20.04 as base

ARG PYTHON_VERSION=3.10
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_HTTP_TIMEOUT=600 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && \
    apt-get install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa &&  \
    apt-get update && \
    apt-get install --no-install-recommends -y \
        python3-pip libpq-dev libgtk2.0-dev libgl1-mesa-dev gcc \
        python${PYTHON_VERSION}-full python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-distutils && \
    rm -rf /var/lib/apt/lists/*

RUN if [ "$PYTHON_VERSION" = "3.8" ]; then \
        python${PYTHON_VERSION} -m pip install --upgrade pip && python${PYTHON_VERSION} -m pip install -U setuptools; \
    else \
        python${PYTHON_VERSION} -m ensurepip --upgrade && python${PYTHON_VERSION} -m pip install --upgrade pip && python${PYTHON_VERSION} -m pip install -U setuptools; \
    fi

RUN python${PYTHON_VERSION} -m pip install uv

ENV PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

COPY ./base /experiment

RUN ln -s /experiment/run.sh /usr/bin/run && \
    chmod +x /experiment/run.sh && \
    chown -R 42420:42420 /experiment
