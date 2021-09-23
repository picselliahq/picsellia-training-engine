docker build --build-arg REBUILD_ALL="$(date +%Y%m%d)" captured-training-tf2/. -t picsellpn/trainingtf2:capture
docker push picsellpn/trainingtf2:capture

docker build --build-arg REBUILD_ALL="$(date +%Y%m%d)" simple-training-tf2/. -t picsellpn/trainingtf2:1.0
docker push picsellpn/trainingtf2:1.0

docker build --build-arg REBUILD_ALL="$(date +%Y%m%d)" simple-training-yolov5/. -t picsellpn/trainingyolov5:1.0
docker push picsellpn/trainingyolov5:1.0

docker build --build-arg REBUILD_ALL="$(date +%Y%m%d)" custom-run/. -t picsellpn/custom-run:1.0
docker push picsellpn/custom-run:1.0

docker build --build-arg REBUILD_ALL="$(date +%Y%m%d)" capture-run/. -t picsellpn/capture-run:1.0
docker push picsellpn/capture-run:1.0