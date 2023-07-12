FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04 as base

COPY ./captured-classification-keras/requirements.txt /requirements.txt

ARG REBUILD_ALL
RUN pip3 install -r ./requirements.txt --no-cache-dir
ARG REBUILD_PICSELLIA

RUN pip3 install --no-cache-dir picsellia

WORKDIR /picsellia

COPY ./captured-classification-keras/picsellia ./

ENTRYPOINT ["run", "main.py"]
