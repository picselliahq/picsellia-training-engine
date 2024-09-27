FROM picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.11

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

ARG REBUILD_ALL
COPY ./src/pipelines/datalake_autotagging/requirements.txt ./src/pipelines/datalake_autotagging/requirements.txt
ARG REBUILD_PICSELLIA

RUN export UV_HTTP_TIMEOUT=900 && uv pip install --python=$(which python3.11) --no-cache -r ./src/pipelines/datalake_autotagging/requirements.txt

WORKDIR /experiment

COPY ./src/decorators ./src/decorators
COPY ./src/models ./src/models
COPY ./src/steps ./src/steps
COPY ./src/*.py ./src
COPY ./src/pipelines/datalake_autotagging ./src/pipelines/datalake_autotagging

RUN mkdir -p /workspace
RUN chmod 777 /workspace

ENV PYTHONPATH=":/experiment/src"

ENTRYPOINT ["run", "python3.11", "src/pipelines/datalake_autotagging/processing_pipeline.py"]
