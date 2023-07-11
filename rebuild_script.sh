docker build captured-training-classif/. -t picsellpn/training-classif:capture
docker push picsellpn/training-classif:capture

docker build captured-classification-keras/. -t picsellpn/classification-keras:capture
docker push picsellpn/classification-keras:capture

docker build simple-training-tf2/. -t picsellpn/trainingtf2:1.0
docker push picsellpn/trainingtf2:1.0

docker build simple-training-yolov5/. -t picsellpn/trainingyolov5:1.0
docker push picsellpn/trainingyolov5:1.0

docker build captured-training-yolov5/. -t picsellpn/trainingyolov5:capture
docker push picsellpn/trainingyolov5:capture

docker build . -f captured-training-tf2/Dockerfile -t picsellpn/trainingtf2:capture
docker push picsellpn/trainingtf2:capture

docker build . -f captured-training-yolov5-detection/Dockerfile -t picsellpn/yolov5-detection:capture
docker push picsellpn/yolov5-detection:capture

docker build . -f captured-training-yolov5-segmentation/Dockerfile -t picsellpn/yolov5-segmentation:capture
docker push picsellpn/yolov5-segmentation:capture

docker build . -f captured-training-yolov8-detection/Dockerfile -t picsellpn/yolov8-detection:capture
docker push picsellpn/yolov8-detection:capture

docker build . -f captured-training-yolov8-segmentation/Dockerfile -t picsellpn/yolov8-segmentation:capture
docker push picsellpn/yolov8-segmentation:capture

docker build . -f captured-training-yolov8-classification/Dockerfile -t picsellpn/yolov8-classification:capture
docker push picsellpn/yolov8-classification:capture