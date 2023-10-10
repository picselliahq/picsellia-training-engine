FROM picsellia/cuda:11.7.1-cudnn8-ubuntu20.04

COPY ./yolov8-segmentation-refactored/requirements.txt .

ARG REBUILD_ALL
RUN python3.10 -m pip install -r requirements.txt --no-cache-dir
ARG REBUILD_PICSELLIA

RUN python3.10 -m pip install picsellia --upgrade

WORKDIR /experiment

COPY ./yolov8-segmentation-refactored/experiment ./
COPY ./evaluator ./evaluator
COPY ./abstract_trainer ./abstract_trainer

ENTRYPOINT ["run", "main.py"]