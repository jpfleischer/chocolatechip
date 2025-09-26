DATASET_URL    := https://www.ccoderun.ca/programming/2024-05-01_LegoGears/legogears_2_dataset.zip
DATASET_DIR    := LegoGears_v2
DATASET_ZIP    := legogears_2_dataset.zip

check-dataset:
	@if [ ! -d $(DATASET_DIR) ]; then \
		echo "Dataset folder '$(DATASET_DIR)' not found. Downloading dataset..."; \
		curl -L $(DATASET_URL) -o $(DATASET_ZIP); \
		unzip -q $(DATASET_ZIP); \
		rm $(DATASET_ZIP); \
		rm LegoGears_v2/LegoGears.cfg && rm LegoGears_v2/LegoGears.data; \
	fi

ultra: check-dataset
	docker pull ultralytics/ultralytics:latest
	docker run -it --shm-size=4gb --gpus all \
		-v $(CURDIR)/LegoGears_v2:/ultralytics/LegoGears_v2 \
		-v $(CURDIR)/LG_v2.yaml:/ultralytics/LG_v2.yaml \
		-v $(CURDIR)/train_setup.py:/ultralytics/train_setup.py \
		ultralytics/ultralytics:latest \
		${CONTAINER_NAME} /bin/bash -c "ls && \
		python3 train_setup.py &&\
		exec /bin/bash"

		# yolo detect train data=LG_v2.yaml model=yolo11n.pt epochs=2134 batch=64

		