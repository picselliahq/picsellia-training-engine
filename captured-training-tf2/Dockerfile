FROM picsellia/cuda:11.4.3-cudnn8-ubuntu20.04

ARG REBUILD_ALL
RUN pip3 install --no-cache-dir picsellia_tf2==0.11.10
ARG REBUILD_PICSELLIA

RUN pip3 install --no-cache-dir picsellia --upgrade

WORKDIR /picsellia

COPY ./captured-training-tf2/picsellia ./
COPY ./evaluator ./evaluator

ENTRYPOINT ["run", "main.py"]