FROM experiment/cuda:11.7.1-cudnn8-ubuntu20.04

COPY ./ViT-classification/requirements.txt .

ARG REBUILD_ALL
RUN pip3 install -r ./requirements.txt --no-cache-dir
ARG REBUILD_PICSELLIA

RUN pip3 install --no-cache-dir picsellia
RUN pip3 install -U accelerate
RUN pip3 install -U transformers

WORKDIR /experiment

COPY ./ViT-classification/experiment ./

ENTRYPOINT ["run", "main.py"]
