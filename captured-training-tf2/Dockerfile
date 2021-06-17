FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04 as base

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt install -y build-essential 
RUN apt-get install --no-install-recommends -y python3.6 python3-pip libpq-dev python3-dev python3-wheel libgtk2.0-dev libgl1-mesa-dev && rm -rf /var/lib/apt/lists/*
RUN python3.6 -m pip install --upgrade pip
RUN pip3 install -U setuptools

COPY ./requirements.txt .
# ARG REBUILD=unknown
RUN pip3 install -r requirements.txt --no-cache-dir
RUN pip3 install --no-cache-dir picsellia
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib:/usr/local/cuda/lib64
COPY ./picsellia /picsellia

CMD ["/picsellia/launch_sub.py"]
ENTRYPOINT ["python3"]
COPY . .
RUN chown -R 42420 /picsellia