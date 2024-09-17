FROM picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.11

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

ARG REBUILD_ALL
COPY ./src/pipelines/datalake_autotagging/requirements.txt ./src/pipelines/datalake_autotagging/requirements.txt
ARG REBUILD_PICSELLIA

RUN export UV_HTTP_TIMEOUT=900 && uv pip install --python=$(which python3.11) --no-cache -r ./src/pipelines/datalake_autotagging/requirements.txt
RUN export UV_HTTP_TIMEOUT=900 && uv pip install --python=$(which python3.11) --no-cache torch==2.2.1+cu118 torchaudio==2.2.1+cu118 torchvision==0.17.1+cu118 --find-links https://download.pytorch.org/whl/torch_stable.html

WORKDIR /experiment

COPY ./src/decorators ./src/decorators
COPY ./src/models ./src/models
COPY ./src/steps ./src/steps
COPY ./src/*.py ./src
COPY ./src/pipelines/datalake_autotagging ./src/pipelines/datalake_autotagging

RUN cd /experiment/src/pipelines/datalake_autotagging && git clone https://github.com/Vision-CAIR/MiniGPT-4 && mv MiniGPT-4 MiniGPT4

ENV PYTHONPATH=":/experiment/src:/experiment/src/pipelines/datalake_autotagging/MiniGPT4"

ENTRYPOINT ["run", "python3.11", "src/pipelines/datalake_autotagging/processing_pipeline.py"]
