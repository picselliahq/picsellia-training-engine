FROM picsellpn/base-11.7.0-cuda:capture

COPY ./captured-training-yolov8-detection/requirements.txt .
ARG REBUILD_ALL
RUN python3.10 -m pip install -r requirements.txt --no-cache-dir --no-cache-dir
ARG REBUILD_PICSELLIA
RUN python3.10 -m pip install picsellia --upgrade

WORKDIR /picsellia

COPY ./captured-training-yolov8-detection/picsellia ./
COPY ./evaluator ./evaluator

ENTRYPOINT ["run", "main.py"]