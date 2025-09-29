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
		-v $(CURDIR)/run.py:/ultralytics/run.py \
		-v $(CURDIR)/hw_info.py:/ultralytics/hw_info.py \
		-v $(CURDIR)/outputs:/ultralytics/outputs \
		ultralytics/ultralytics:latest \
		${CONTAINER_NAME} /bin/bash -c "ls && \
		python3 train_setup.py &&\
		exec /bin/bash"

benchmark-ultra benchmark: check-dataset
	docker pull ultralytics/ultralytics:latest
	docker run -it --shm-size=4gb --gpus all \
		-v $(CURDIR)/LegoGears_v2:/ultralytics/LegoGears_v2 \
		-v $(CURDIR)/LG_v2.yaml:/ultralytics/LG_v2.yaml \
		-v $(CURDIR)/train_setup.py:/ultralytics/train_setup.py \
		-v $(CURDIR)/run.py:/ultralytics/run.py \
		-v $(CURDIR)/hw_info.py:/ultralytics/hw_info.py \
		-v $(CURDIR)/outputs:/ultralytics/outputs \
		ultralytics/ultralytics:latest \
		${CONTAINER_NAME} /bin/bash -c "ls && \
		python3 train_setup.py &&\
		pip install --no-cache-dir cloudmesh-common git+https://github.com/cloudmesh/cloudmesh-gpu.git &&\
		python3 run.py &&\
		exec /bin/bash"

# yolo detect train data=LG_v2.yaml model=yolo11n.pt epochs=2134 batch=64
# yolo detect train data=LG_v2.yaml model=yolo11n.pt epochs=1 batch=64
#moving files to get the info : mv runs/detect/train outputs
#locate benchmark_bundle__default_username__NVIDIA_RTX_A6000-_NVIDIA_RTX_A6000__Intel_R_Xeon_R_Gold_6226R_CPU_2.90GHz__20250928_160849.zip

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

		