FROM picsellpn/base-11.7.0-cuda:capture

COPY ./captured-training-yolov8-classification/requirements.txt .
RUN python3.10 -m pip install -r requirements.txt

WORKDIR /picsellia

COPY ./captured-training-yolov8-classification/picsellia ./
COPY ./evaluator ./evaluator

ENTRYPOINT ["./run.sh", "/picsellia/main.py"]