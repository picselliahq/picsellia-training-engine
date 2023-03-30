docker build --build-arg REBUILD_ALL="$(date +%Y%m%d%h)" captured-training-tf2/. -t picsellpn/trainingtf2:capture
docker push picsellpn/trainingtf2:capture

docker build --build-arg REBUILD_ALL="$(date +%Y%m%d%h)" captured-training-classif/. -t picsellpn/training-classif:capture
docker push picsellpn/training-classif:capture

docker build --build-arg REBUILD_ALL="$(date +%Y%m%d%h)" captured-classification-keras/. -t picsellpn/classification-keras:capture
docker push picsellpn/classification-keras:capture

docker build --build-arg REBUILD_ALL="$(date +%Y%m%d%h)" simple-training-tf2/. -t picsellpn/trainingtf2:1.0
docker push picsellpn/trainingtf2:1.0

# docker build --build-arg REBUILD_ALL="$(date +%Y%m%d%h)" simple-training-yolov5/. -t picsellpn/trainingyolov5:1.0
# docker push picsellpn/trainingyolov5:1.0

docker build --build-arg REBUILD_ALL="$(date +%Y%m%d%h)" captured-training-yolov5/. -t picsellpn/trainingyolov5:capture
docker push picsellpn/trainingyolov5:capture

docker build captured-training-yolov5-detection/. -t picsellpn/yolov5-detection:capture
docker push picsellpn/yolov5-detection:capture

docker build captured-training-yolov5-segmentation/. -t picsellpn/yolov5-segmentation:capture
docker push picsellpn/yolov5-segmentation:capture

docker build captured-training-yolov8-detection/. -t picsellpn/yolov8-detection:capture
docker push picsellpn/yolov8-detection:capture

docker build captured-training-yolov8-segmentation/. -t picsellpn/yolov8-segmentation:capture
docker push picsellpn/yolov8-segmentation:capture