docker build . -f base/11.4.3-cuda-cudnn8/Dockerfile -t picsellia/cuda:11.4.3-cudnn8-ubuntu20.04
docker push picsellia/cuda:11.4.3-cudnn8-ubuntu20.04

docker build . -f base/11.7.1-cuda-cudnn8/Dockerfile -t picsellia/cuda:11.7.1-cudnn8-ubuntu20.04
docker push picsellia/cuda:11.7.1-cudnn8-ubuntu20.04

docker build . -f captured-training-tf2/Dockerfile -t picsellia/training-tf2:latest
docker push picsellia/training-tf2:latest

docker build . -f captured-classification-keras/Dockerfile -t picsellia/training-keras-classification:latest
docker push picsellia/training-keras-classification:latest

docker build . -f captured-training-yolov5-detection/Dockerfile -t picsellia/training-yolov5-detection:latest
docker push picsellia/training-yolov5-detection:latest

docker build . -f captured-training-yolov5-segmentation/Dockerfile -t picsellia/training-yolov5-segmentation:latest
docker push picsellia/training-yolov5-segmentation:latest

docker build . -f captured-training-yolov8-classification/Dockerfile -t picsellia/training-yolov8-classification:latest
docker push picsellia/training-yolov8-classification:latest

docker build . -f captured-training-yolov8-detection/Dockerfile -t picsellia/training-yolov8-detection:latest
docker push picsellia/training-yolov8-detection:latest

docker build . -f captured-training-yolov8-segmentation/Dockerfile -t picsellia/training-yolov8-segmentation:latest
docker push picsellia/training-yolov8-segmentation:latest