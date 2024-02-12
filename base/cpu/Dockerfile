ARG PYTHON_VERSION=3.8

FROM python:${PYTHON_VERSION}-bookworm as base

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ARG DEBIAN_FRONTEND=noninteractive

RUN pip3 install --upgrade pip
RUN pip3 install -U setuptools

COPY ./base /experiment

RUN ln -s /experiment/run.sh /usr/bin/run

RUN chmod +x /experiment/run.sh
RUN chown -R 42420:42420 /experiment