ARG PYTHON_VERSION=3.8

FROM python:${PYTHON_VERSION}-bookworm as base

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ARG DEBIAN_FRONTEND=noninteractives

RUN if [ "$PYTHON_VERSION" = "3.10" ]; then \
        python${PYTHON_VERSION} -m ensurepip --upgrade && python${PYTHON_VERSION} -m pip install -U setuptools; \
    fi

RUN python${PYTHON_VERSION} -m pip install uv

COPY ./base /experiment

RUN ln -s /experiment/run.sh /usr/bin/run && \
    chmod +x /experiment/run.sh && \
    chown -R 42420:42420 /experiment
