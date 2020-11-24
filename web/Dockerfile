FROM ubuntu:18.04

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install --no-install-recommends -y python3.6 python3-pip python3-setuptools curl libpq-dev python3-dev python3-wheel libgtk2.0-dev libsm6 libxext6 libxrender-dev build-essential cmake && rm -rf /var/lib/apt/lists/*
RUN apt-get update
RUN apt-get install -y gnupg wget
RUN apt-get install -y apt-transport-https ca-certificates 
COPY app.py ./
COPY wsgi.py ./
COPY requirements_server.txt ./

RUN pip3 install -r requirements_server.txt

RUN add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
RUN apt-get update 
RUN apt-get install docker-ce docker-ce-cli
