FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04 as base

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt install -y build-essential 
RUN apt-get install --no-install-recommends -y python3.6 python3-pip libpq-dev python3-dev python3-wheel libgtk2.0-dev libgl1-mesa-dev && rm -rf /var/lib/apt/lists/*
RUN python3.6 -m pip install --upgrade pip
RUN pip3 install -U setuptools

COPY ./requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install --no-cache-dir picsellia_training==0.0.9
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib:/usr/local/cuda/lib64
ADD start.sh /

RUN chmod +x /start.sh
ENTRYPOINT ["/start.sh"]
COPY . .