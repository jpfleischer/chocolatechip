run: build
	docker run --gpus all \
	-v /data/:/data/ \
	-v ${CURDIR}/output:/workspace/output \
	yolo_converter


build:
	docker build -t yolo_converter .
