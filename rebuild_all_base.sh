#!/bin/bash

declare -A cuda_versions_python=(
    ["11.4.3"]="3.8"
    ["11.7.1"]="3.10"
    ["11.8.0"]="3.10 3.11 3.12"
)

for cuda_version in "${!cuda_versions_python[@]}"; do
    for python_version in ${cuda_versions_python[$cuda_version]}; do
        image_tag="picsellia/cuda:${cuda_version}-cudnn8-ubuntu20.04-python${python_version}"
        docker build --build-arg CUDA_VERSION=$cuda_version --build-arg PYTHON_VERSION=$python_version . -f base/cuda/Dockerfile -t $image_tag
        docker push $image_tag
    done
done

python_versions_cpu=("3.10" "3.11" "3.12")

for python_version in "${python_versions_cpu[@]}"; do
    image_tag="picsellia/cpu:ubuntu20.04-python${python_version}"
    docker build --build-arg PYTHON_VERSION=$python_version . -f base/cpu/Dockerfile -t $image_tag
    docker push $image_tag
done
