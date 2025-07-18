run: build
	docker run \
	-v /data/:/data/ \
	-v ${CURDIR}/output:/workspace/output \
	yolo_converter


build:
	docker build -t yolo_converter .
