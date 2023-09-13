FROM experiment/cuda:11.7.1-cudnn8-ubuntu20.04

COPY ./yolov8-classification-old/requirements.txt .

ARG REBUILD_ALL
RUN python3.10 -m pip install -r requirements.txt --no-cache-dir
ARG REBUILD_PICSELLIA

RUN python3.10 -m pip install picsellia --upgrade

WORKDIR /experiment

COPY ./yolov8-classification-old/picsellia ./
COPY ./evaluator ./evaluator

ENTRYPOINT ["run", "main.py"]