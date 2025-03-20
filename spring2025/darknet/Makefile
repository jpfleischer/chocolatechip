CONTAINER_NAME := my_darknet_container
COLLAB_DIR     := ${CURDIR}/../cvat/docker/
COLLABFILES_DIR:= ${CURDIR}/CollabFiles
DATASET_DIR    := docker/LegoGears_v2
EXTRAFILES_DIR := LegoGearsFiles
DATASET_ZIP    := legogears_2_dataset.zip
DATASET_URL    := https://www.ccoderun.ca/programming/2024-05-01_LegoGears/legogears_2_dataset.zip
# Name for the apptainer image file
SIF_NAME       := my_darknet_container.sif
# Local directory for a writable workspace
DARKNET_WORKSPACE := $(CURDIR)/darknet_workspace
# Bind mount option stored in a variable
BIND_OPT := -B $(DARKNET_WORKSPACE):/host_workspace:rw


.PHONY: check-dataset build run stop apptainer-build arun ashell

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



# Build the apptainer image using the definition file
$(SIF_NAME): apptainer.def
	apptainer build --force $(SIF_NAME) apptainer.def

apptainer-build: $(SIF_NAME)

# Create the workspace directory if it doesn't exist
$(DARKNET_WORKSPACE):
	mkdir -p $(DARKNET_WORKSPACE)

delete:
	rm -rf ./docker
	rm -rf ./darknet_workspace
	-rm darknet_train.out darknet_train.err

# Run the apptainer container with a bind mount for a writable workspace
arun: check-dataset apptainer-build $(DARKNET_WORKSPACE) arun-command
	
arun-command:
	apptainer run --nv --fakeroot \
		-B $(DARKNET_WORKSPACE):/host_workspace \
		-B ${CURDIR}/docker:/workspace \
		-B ${CURDIR}/LegoGearsFiles/outputs:/outputs \
		$(SIF_NAME)

arun-cms-gpu:
	apptainer exec --nv --fakeroot \
		-B $(DARKNET_WORKSPACE):/host_workspace \
		-B ${CURDIR}/docker:/workspace \
		-B ${CURDIR}/LegoGearsFiles/outputs:/outputs \
		$(SIF_NAME) cms gpu watch --gpu=0 --delay=1 --dense > gpu0new.log 2>&1 &

arun-cars-command:
	apptainer run --nv --fakeroot \
		-B $(DARKNET_WORKSPACE):/host_workspace \
		-B ${CURDIR}/CollabFiles:/workspace \
		$(SIF_NAME)

ashell:
	apptainer shell $(SIF_NAME)

slurm: delete check-dataset apptainer-build $(DARKNET_WORKSPACE)
	sbatch -J "lego_train_$(date +%Y%m%d_%H%M%S)" legogears.slurm
	watch squeue -u $(USER)

slurm-shell:
	srun -p gpu --gpus=geforce:1 --time=01:00:00 --mem=50gb --pty apptainer shell --nv $(SIF_NAME)


cars-slurm: delete check-dataset apptainer-build $(DARKNET_WORKSPACE)
	sbatch -J "darknet_cars_train_$(date +%Y%m%d_%H%M%S)" CollabFiles/cars.slurm