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

# docker build --build-arg REBUILD_ALL="$(date +%Y%m%d%h)" custom-run/. -t picsellpn/custom-run:1.0
# docker push picsellpn/custom-run:1.0

# docker build --build-arg REBUILD_ALL="$(date +%Y%m%d%h)" capture-run/. -t picsellpn/capture-run:1.0
# docker push picsellpn/capture-run:1.0