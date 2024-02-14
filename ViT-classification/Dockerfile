FROM picsellia/cuda:11.7.1-cudnn8-ubuntu20.04-python3.10

COPY ./ViT-classification/requirements.txt .

ARG REBUILD_ALL
RUN pip3 install -r ./requirements.txt --no-cache-dir
ARG REBUILD_PICSELLIA


RUN pip3 install --no-cache-dir picsellia
RUN pip3 install picsellia --upgrade

RUN pip3 install -U accelerate
RUN pip3 install -U transformers
RUN pip3 install optimum[onnxruntime]

WORKDIR /experiment
COPY ./ViT-classification/experiment ./
RUN mkdir -p /workspace
COPY ./abstract_trainer ./abstract_trainer
COPY ./evaluator ./evaluator
RUN chown -R 42420:42420 /workspace

ENTRYPOINT ["run", "main.py"]
