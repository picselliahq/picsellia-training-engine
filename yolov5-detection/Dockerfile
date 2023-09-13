FROM experiment/cuda:11.4.3-cudnn8-ubuntu20.04

COPY ./yolov5-detection/requirements.txt .

ARG REBUILD_ALL
RUN pip3 install -r ./requirements.txt --no-cache-dir
ARG REBUILD_PICSELLIA

RUN python3 -m pip install picsellia-yolov5==0.2.8
RUN pip3 install picsellia --upgrade

WORKDIR /experiment

COPY ./yolov5-detection/experiment ./

ENTRYPOINT ["run", "main.py"]