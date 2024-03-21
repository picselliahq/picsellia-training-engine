FROM picsellia/cuda:11.4.3-cudnn8-ubuntu20.04-python3.8

COPY ./keras-classification/requirements.txt /requirements.txt

ARG REBUILD_ALL
RUN python3.8 -m pip install -r ./requirements.txt --no-cache-dir
ARG REBUILD_PICSELLIA

RUN python3.8 -m pip install --no-cache-dir picsellia

WORKDIR /experiment

COPY ./keras-classification/experiment ./

ENTRYPOINT ["run", "main.py"]
