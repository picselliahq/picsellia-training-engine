FROM picsellpn/base-11.4.2-cuda:capture

COPY ./captured-training-yolov7/requirements.txt .
ARG REBUILD_ALL
RUN pip3 install -r requirements.txt --no-cache-dir
ARG REBUILD_PICSELLIA
RUN pip3 install picsellia --upgrade

WORKDIR /picsellia

COPY ./captured-training-yolov7/picsellia ./

ENTRYPOINT ["run", "main.py"]