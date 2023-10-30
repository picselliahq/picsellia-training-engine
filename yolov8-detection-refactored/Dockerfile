FROM picsellia/cuda:11.7.1-cudnn8-ubuntu20.04

COPY ./yolov8-detection-refactored/requirements.txt .

ARG REBUILD_ALL
RUN python3.10 -m pip install -r requirements.txt --no-cache-dir

WORKDIR /experiment

COPY ./yolov8-detection-refactored/experiment ./
COPY ./evaluator ./evaluator
COPY ./abstract_trainer ./abstract_trainer
COPY ./core_utils ./core_utils

ENTRYPOINT ["run", "main.py"]