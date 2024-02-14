FROM picsellia/cuda:11.7.1-cudnn8-ubuntu20.04-python3.10

COPY ./yolox-detection/requirements.txt .

ARG REBUILD_ALL
RUN pip3 install -r requirements.txt --no-cache-dir

WORKDIR /experiment

COPY ./yolox-detection/experiment ./
COPY ./evaluator ./evaluator

RUN mkdir -p /workspace
RUN chmod 777 /workspace

ENTRYPOINT ["run", "main.py"]