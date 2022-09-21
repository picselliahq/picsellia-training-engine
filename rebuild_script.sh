docker build captured-training-tf2/. -t picsellpn/trainingtf2:capture
docker push picsellpn/trainingtf2:capture

docker build captured-training-classif/. -t picsellpn/training-classif:capture
docker push picsellpn/training-classif:capture

docker build simple-training-tf2/. -t picsellpn/trainingtf2:1.0
docker push picsellpn/trainingtf2:1.0

docker build simple-training-yolov5/. -t picsellpn/trainingyolov5:1.0
docker push picsellpn/trainingyolov5:1.0

docker build captured-training-yolov5/. -t picsellpn/trainingyolov5:capture
docker push picsellpn/trainingyolov5:capture

docker build custom-run/. -t picsellpn/custom-run:1.0
docker push picsellpn/custom-run:1.0

docker build capture-run/. -t picsellpn/capture-run:1.0
docker push picsellpn/capture-run:1.0