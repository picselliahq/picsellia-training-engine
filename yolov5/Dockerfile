FROM experiment/base-11.4.3-cuda:capture

COPY ./yolov5/requirements.txt .

ARG REBUILD_ALL
RUN pip3 install -r ./requirements.txt --no-cache-dir
ARG REBUILD_PICSELLIA

RUN python3 -m pip install experiment-yolov5==0.2.8
RUN pip3 install experiment --upgrade

WORKDIR /experiment

COPY ./yolov5/picsellia ./

ENTRYPOINT ["run", "main.py"]