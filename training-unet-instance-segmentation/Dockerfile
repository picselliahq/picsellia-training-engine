FROM picsellia/cuda:11.7.1-cudnn8-ubuntu20.04

COPY ./training-unet-instance-segmentation/requirements.txt .

ARG REBUILD_ALL
RUN pip3 install -r ./requirements.txt --no-cache-dir
ARG REBUILD_PICSELLIA

RUN pip3 install --no-cache-dir picsellia

WORKDIR /picsellia
ENV SM_FRAMEWORK="tf.keras"

COPY ./training-unet-instance-segmentation/modeling ./
COPY ./abstract_trainer ./abstract_trainer

ENTRYPOINT ["run", "main.py"]
