FROM picsellia/cuda:11.7.1-cudnn8-ubuntu20.04-python3.10

COPY ./yolov9-segmentation/requirements.txt .

ARG REBUILD_ALL
RUN python3.10 -m pip install -r requirements.txt --no-cache-dir

WORKDIR /experiment

COPY ./yolov9-segmentation/experiment ./
COPY ./evaluator ./evaluator
COPY ./abstract_trainer ./abstract_trainer
COPY ./core_utils ./core_utils

RUN mkdir -p /workspace
RUN chmod 777 /workspace

ENTRYPOINT ["run", "python3.10", "main.py"]
