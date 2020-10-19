#!/bin/sh

ln -s /usr/local/cuda/lib64/libcudart.so.10.2 /usr/local/cuda/lib64/libcudart.so.10.0
ln -s /usr/lib/x86_64-linux-gnu/libcublas.so.10 /usr/local/cuda/lib64/libcublas.so.10.0
ln -s /usr/local/cuda/lib64/libcufft.so.10 /usr/local/cuda/lib64/libcufft.so.10.0
ln -s /usr/local/cuda/lib64/libcurand.so.10 /usr/local/cuda/lib64/libcurand.so.10.0
ln -s /usr/local/cuda/lib64/libcusolver.so.10 /usr/local/cuda/lib64/libcusolver.so.10.0
ln -s /usr/local/cuda/lib64/libcusparse.so.10 /usr/local/cuda/lib64/libcusparse.so.10.0

python3 launch_sub.py 

