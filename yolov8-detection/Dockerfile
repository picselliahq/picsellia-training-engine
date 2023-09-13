FROM picsellia/base-11.4.3-cuda:capture

COPY ./yolov8-detection/requirements.txt .

ARG REBUILD_ALL
RUN python3.10 -m pip install -r requirements.txt --no-cache-dir
ARG REBUILD_PICSELLIA

RUN python3.10 -m pip install experiment --upgrade

WORKDIR /experiment

COPY ./yolov8-detection/experiment ./
COPY ./evaluator ./evaluator

ENTRYPOINT ["run", "main.py"]