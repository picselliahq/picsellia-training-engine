ARG PYTHON_VERSION=3.8
FROM python:${PYTHON_VERSION}-bookworm AS base

ARG PYTHON_VERSION=3.8

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_HTTP_TIMEOUT=600 \
    DEBIAN_FRONTEND=noninteractive

RUN python${PYTHON_VERSION} -m pip install --upgrade pip setuptools && \
    python${PYTHON_VERSION} -m pip install uv && \
    uv pip install --python=$(which python${PYTHON_VERSION}) --no-cache-dir tabulate

COPY ./base /experiment

RUN ln -s /experiment/run.sh /usr/bin/run && \
    chmod +x /experiment/run.sh && \
    chown -R 42420:42420 /experiment
