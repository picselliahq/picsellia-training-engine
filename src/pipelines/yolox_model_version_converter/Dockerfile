FROM picsellia/cpu:ubuntu20.04-python3.11

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

ARG REBUILD_ALL
COPY ./src/pipelines/model_version_converter/requirements.txt ./src/pipelines/model_version_converter/requirements.txt
ARG REBUILD_PICSELLIA

RUN uv pip install --python=$(which python3.11) --no-cache -r ./src/pipelines/model_version_converter/requirements.txt

WORKDIR /experiment

COPY ./src/decorators ./src/decorators
COPY ./src/models ./src/models
COPY ./src/steps ./src/steps
COPY ./src/*.py ./src
COPY ./src/pipelines/model_version_converter ./src/pipelines/model_version_converter

RUN mkdir -p /workspace
RUN chmod 777 /workspace

ENV PYTHONPATH "${PYTHONPATH}:/experiment/src"

ENTRYPOINT ["run", "python3.11", "src/pipelines/model_version_converter/processing_pipeline.py"]
