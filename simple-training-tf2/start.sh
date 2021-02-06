#!/bin/sh

sudo ln -s /usr/local/cuda/lib64/libcudart.so.10.2 /usr/local/cuda/lib64/libcudart.so.10.0
sudo ln -s /usr/lib/x86_64-linux-gnu/libcublas.so.10 /usr/local/cuda/lib64/libcublas.so.10.0
sudo ln -s /usr/local/cuda/lib64/libcufft.so.10 /usr/local/cuda/lib64/libcufft.so.10.0
sudo ln -s /usr/local/cuda/lib64/libcurand.so.10 /usr/local/cuda/lib64/libcurand.so.10.0
sudo ln -s /usr/local/cuda/lib64/libcusolver.so.10 /usr/local/cuda/lib64/libcusolver.so.10.0
sudo ln -s /usr/local/cuda/lib64/libcusparse.so.10 /usr/local/cuda/lib64/libcusparse.so.10.0

# gunicorn --bind 0.0.0.0:5000 -w 4 -t 360 wsgi:app
python3 docker_run_training_tf2.py

