CONTAINER_NAME := my_darknet_container
COLLAB_DIR     := ${CURDIR}/../cvat/docker/
COLLABFILES_DIR:= ${CURDIR}/CollabFiles
DATASET_DIR    := docker/LegoGears_v2
EXTRAFILES_DIR := LegoGearsFiles
DATASET_ZIP    := legogears_2_dataset.zip
DATASET_URL    := https://www.ccoderun.ca/programming/2024-05-01_LegoGears/legogears_2_dataset.zip

.PHONY: check-dataset build run stop


run: check-dataset build
	@docker run --rm -it --gpus all --name $(CONTAINER_NAME) -v ${CURDIR}/docker:/workspace darknet-container


collab: check-dataset-collab build
	@docker run --rm -it --gpus all --name $(CONTAINER_NAME) -v $(COLLAB_DIR):/workspace darknet-container

check-dataset:
	@if [ ! -d $(DATASET_DIR) ]; then \
		mkdir -p docker; \
		echo "Dataset folder '$(DATASET_DIR)' not found. Downloading dataset..."; \
		curl -L $(DATASET_URL) -o $(DATASET_ZIP); \
		unzip -q $(DATASET_ZIP) -d docker; \
		rm $(DATASET_ZIP); \
		rm $(DATASET_DIR)/LegoGears.cfg && rm $(DATASET_DIR)/LegoGears.data; \
		cp $(EXTRAFILES_DIR)/LegoGears.cfg $(DATASET_DIR)/LegoGears.cfg; \
		cp $(EXTRAFILES_DIR)/LegoGears.data $(DATASET_DIR)/LegoGears.data; \
		cp $(EXTRAFILES_DIR)/train_setup.py docker/train_setup.py; \
		cp $(EXTRAFILES_DIR)/run.py docker/run.py; \
		cp $(EXTRAFILES_DIR)/script.sh docker/script.sh; \
	fi


check-dataset-collab:
	cp $(COLLABFILES_DIR)/cars.cfg $(COLLAB_DIR)/cars.cfg;
	cp $(COLLABFILES_DIR)/cars.data $(COLLAB_DIR)/cars.data;
	cp $(COLLABFILES_DIR)/cars.names $(COLLAB_DIR)/cars.names;
	cp $(COLLABFILES_DIR)/train_setup.py $(COLLAB_DIR)/train_setup.py;
	cp $(COLLABFILES_DIR)/run.py $(COLLAB_DIR)/run.py;
	cp $(COLLABFILES_DIR)/script.sh $(COLLAB_DIR)/script.sh;

build:
	@docker build -t darknet-container .

shell:
	@docker exec -it $(CONTAINER_NAME) /bin/bash

stop:
	@-docker stop $(CONTAINER_NAME)
	@-docker rm $(CONTAINER_NAME)
