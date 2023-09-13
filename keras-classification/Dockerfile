FROM experiment/cuda:11.4.3-cudnn8-ubuntu20.04

COPY ./keras-classification/requirements.txt /requirements.txt

ARG REBUILD_ALL
RUN pip3 install -r ./requirements.txt --no-cache-dir
ARG REBUILD_PICSELLIA

RUN pip3 install --no-cache-dir picsellia

WORKDIR /experiment

COPY ./keras-classification/experiment ./

ENTRYPOINT ["run", "main.py"]
