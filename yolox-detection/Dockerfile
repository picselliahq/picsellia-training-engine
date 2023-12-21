FROM picsellia/cuda:11.7.1-cudnn8-ubuntu20.04

COPY ./yolox-detection/requirements.txt .

ARG REBUILD_ALL
RUN python3.10 -m pip install -r requirements.txt --no-cache-dir

RUN git clone git@github.com:Megvii-BaseDetection/YOLOX.git
RUN pip3 install -U pip && pip3 install -r YOLOX/requirements.txt
RUN python3 YOLOX/setup.py develop


WORKDIR /experiment

COPY ./yolox-detection/experiment ./
COPY ./evaluator ./evaluator
COPY ./abstract_trainer ./abstract_trainer
COPY ./core_utils ./core_utils

ENTRYPOINT ["run", "main.py"]