FROM picsellia/cuda:11.4.3-cudnn8-ubuntu20.04-python3.8

ARG REBUILD_ALL
RUN python3.8 -m pip install --no-cache-dir picsellia_tf2==0.11.10
ARG REBUILD_PICSELLIA

RUN python3.8 -m pip install --no-cache-dir picsellia --upgrade

WORKDIR /experiment

COPY ./tf2/experiment ./
COPY ./evaluator ./evaluator

ENTRYPOINT ["run", "main.py"]