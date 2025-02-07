CONTAINER_NAME := my_darknet_container
DATASET_DIR    := LegoGears_v2
DATASET_ZIP    := legogears_2_dataset.zip
DATASET_URL    := https://www.ccoderun.ca/programming/2024-05-01_LegoGears/legogears_2_dataset.zip

.PHONY: check-dataset build run stop


run: check-dataset build
	@docker run --rm -it --gpus all --name $(CONTAINER_NAME) darknet-container

check-dataset:
	@if [ ! -d $(DATASET_DIR) ]; then \
		echo "Dataset folder '$(DATASET_DIR)' not found. Downloading dataset..."; \
		wget $(DATASET_URL) -O $(DATASET_ZIP); \
		unzip $(DATASET_ZIP); \
		rm $(DATASET_ZIP); \
	fi

build: check-dataset
	@docker build -t darknet-container .


stop:
	@-docker stop $(CONTAINER_NAME)
	@-docker rm $(CONTAINER_NAME)
