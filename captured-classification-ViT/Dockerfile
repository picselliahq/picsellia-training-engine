FROM picsellia/cuda:11.4.3-cudnn8-ubuntu20.04

COPY ./captured-classification-ViT/requirements.txt .

ARG REBUILD_ALL
RUN pip3 install -r ./requirements.txt --no-cache-dir
ARG REBUILD_PICSELLIA

RUN pip3 install --no-cache-dir picsellia

WORKDIR /picsellia

COPY ./captured-classification-ViT/picsellia ./

ENTRYPOINT ["run", "main.py"]
