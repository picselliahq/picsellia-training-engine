FROM picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.10

COPY ./unet-instance-segmentation/requirements.txt .

ARG REBUILD_ALL
RUN pip3 install -r ./requirements.txt --no-cache-dir
RUN pip3 install picsellia --upgrade

WORKDIR /experiment
ENV SM_FRAMEWORK="tf.keras"

COPY ./unet-instance-segmentation/experiment ./
COPY ./abstract_trainer ./abstract_trainer

ENTRYPOINT ["run", "main.py"]
