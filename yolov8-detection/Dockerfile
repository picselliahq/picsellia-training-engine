FROM picsellia/cuda:11.7.1-cudnn8-ubuntu20.04-python3.10

COPY ./yolov8-detection/requirements.txt .

ARG REBUILD_ALL
RUN pip3 install -r requirements.txt --no-cache-dir

WORKDIR /experiment

COPY ./yolov8-detection/experiment ./
COPY ./evaluator ./evaluator
COPY ./abstract_trainer ./abstract_trainer
COPY ./core_utils ./core_utils

ENTRYPOINT ["run", "main.py"]