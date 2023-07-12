FROM picsellpn/base-11.4.3-cuda:capture

COPY ./captured-training-yolov5-segmentation/requirements.txt .
ARG REBUILD_ALL
RUN pip3 install -r ./requirements.txt --no-cache-dir
ARG REBUILD_PICSELLIA
RUN python3 -m pip install picsellia-yolov5==0.2.8
RUN pip3 install picsellia --upgrade

WORKDIR /picsellia

COPY ./captured-training-yolov5-segmentation/picsellia ./

ENTRYPOINT ["run", "main.py"]