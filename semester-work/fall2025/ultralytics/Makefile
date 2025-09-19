yolo-ultra:
	docker pull ultralytics/ultralytics:latest
	docker run -it --gpus all \
		-v $(CURDIR)/LegoGears:/ultralytics/LegoGears \
		-v $(CURDIR)/LegoGears_v2:/ultralytics/LegoGears_v2 \
		-v $(CURDIR)/LG_v2.yaml:/ultralytics/LG_v2.yaml \
		ultralytics/ultralytics:latest \
		${CONTAINER_NAME} /bin/bash -c "ls && \
		exec /bin/bash"

		# yolo detect train data=LG_v2.yaml model=yolo11n.pt epochs=1 

		