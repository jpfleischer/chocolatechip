run: build
	docker run --gpus all \
	-v /data/traffic_training_cvat_unzipped/annotations:/data/traffic_training_cvat_unzipped/annotations \
	-v ${CURDIR}/output:/workspace/output \
	yolo_converter


build:
	docker build -t yolo_converter .
