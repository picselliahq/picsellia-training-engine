FROM picsellia/cpu:ubuntu20.04-python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

ARG REBUILD_ALL
COPY ./src/pipelines/slicer/requirements.txt ./src/pipelines/slicer/requirements.txt
ARG REBUILD_PICSELLIA

RUN uv pip install --python=$(which python3.10) --no-cache -r ./src/pipelines/slicer/requirements.txt

WORKDIR /experiment

COPY ./src/decorators ./src/decorators
COPY ./src/models ./src/models
COPY ./src/steps ./src/steps
COPY ./src/*.py ./src
COPY ./src/pipelines/slicer ./src/pipelines/slicer

ENV PYTHONPATH "${PYTHONPATH}:/experiment/src"

ENTRYPOINT ["run", "python3.10", "src/pipelines/slicer/processing_pipeline.py"]
