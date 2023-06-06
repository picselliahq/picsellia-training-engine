FROM picsellpn/base-11.7.0-cuda:capture

COPY ./captured-training-yolov8-segmentation/requirements.txt .
ARG REBUILD_ALL
RUN python3.10 -m pip install -r requirements.txt --no-cache-dir
ARG REBUILD_PICSELLIA
RUN python3.10 -m pip install picsellia --upgrade

WORKDIR /picsellia

COPY ./captured-training-yolov8-segmentation/picsellia ./
COPY ./evaluator ./evaluator

ENTRYPOINT ["run", "main.py"]