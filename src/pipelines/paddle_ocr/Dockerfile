FROM picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

ARG REBUILD_ALL
COPY ./src/pipelines/paddle_ocr/requirements.txt ./src/pipelines/paddle_ocr/requirements.txt
ARG REBUILD_PICSELLIA

RUN uv pip install --python=$(which python3.10) --no-cache -r ./src/pipelines/paddle_ocr/requirements.txt

RUN git clone --branch release/2.6 https://github.com/PaddlePaddle/PaddleOCR

WORKDIR /experiment

COPY ./src/decorators ./src/decorators
COPY ./src/models ./src/models
COPY ./src/steps ./src/steps
COPY ./src/*.py ./src
COPY ./src/pipelines/paddle_ocr ./src/pipelines/paddle_ocr

ENV PYTHONPATH "${PYTHONPATH}:/experiment/src"

ENTRYPOINT ["run", "python3.10", "src/pipelines/paddle_ocr/processing_pipeline.py"]
