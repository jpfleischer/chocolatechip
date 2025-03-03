run: build
	docker run --gpus all -v /data:/data -v $(PWD)/output:/workspace/tensorrt_demos/yolo/output yolo_converter

build:
	docker build -t yolo_converter .
