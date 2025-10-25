CONTAINER_NAME := my_darknet_container
COLLAB_DIR     := ${CURDIR}/../cvat/docker/
COLLABFILES_DIR:= ${CURDIR}/CollabFiles
SIF_NAME       := $(CURDIR)/my_darknet_container.sif
# Local directory for a writable workspace
DARKNET_WORKSPACE := $(CURDIR)/darknet_workspace
# Bind mount option stored in a variable
BIND_OPT := -B $(DARKNET_WORKSPACE):/host_workspace:rw
PROJ_ROOT := $(shell git -C "$(CURDIR)" rev-parse --show-toplevel)

YOLO_TEMPLATES ?= all   # values: all | yolov4-tiny | yolov7-tiny

# Determine if we're on a Virginia host.
UVA_VIRGINIA_RUNNING := $(shell \
	if echo "$$(dig -x $$(dig +short myip.opendns.com @resolver1.opendns.com) +short)" | grep -q "virginia.edu"; then \
		echo true; \
	else \
		echo false; \
	fi)

# Define a variable for Docker to pass the UVA environment variable.
DOCKER_UVA_ENV := --env UVA_VIRGINIA_RUNNING=$(UVA_VIRGINIA_RUNNING)

# Define a variable to conditionally load apptainer if at UVA.
MODULE_LOAD_APPTAINER = if [ "$(UVA_VIRGINIA_RUNNING)" = "true" ]; then \
		. /etc/profile.d/modules.sh && module load apptainer; \
	fi

ACTUAL_PWD := $(shell pwd)

ifeq ($(OS),Windows_NT)
  # Check for Git Bash by inspecting uname output for "MINGW"
  GIT_BASH_CHECK := $(shell uname -s 2>/dev/null)
  ifeq ($(findstring MINGW,$(GIT_BASH_CHECK)),)
    $(error "This Makefile must be run from Git Bash on Windows. Detected: '$(GIT_BASH_CHECK)'")
  endif
  WINDOWS_HD := $(shell wmic diskdrive where "DeviceID like '%PHYSICALDRIVE0%'" get Model | grep -v Model | sed 's/^[[:space:]]*//' | sed 's/\r//')
  WINDOWS_HD := $(strip $(WINDOWS_HD))
  WINDOWS_CAP := $(shell wmic diskdrive where "DeviceID like '%PHYSICALDRIVE0%'" get Size | grep -v Size | sed 's/^[[:space:]]*//' | sed 's/\r//')
  WINDOWS_CAP := $(strip $(WINDOWS_CAP))
  # Set both environment variables for Docker.
  DOCKER_WINDOWS_ENV := --env WINDOWS_HARD_DRIVE="$(WINDOWS_HD)" --env WINDOWS_HARD_DRIVE_CAPACITY="$(WINDOWS_CAP)"
  # --- Build IDs on Git Bash/Windows: stick to 1000:1000 like your working setup ---
  BUILD_UID := 1000
  BUILD_GID := 1000
else
  DOCKER_WINDOWS_ENV :=
  # --- Build IDs on Linux/macOS: match the current user ---
  BUILD_UID := $$(id -u)
  BUILD_GID := $$(id -g)
endif


.PHONY: build run stop apptainer-build arun ashell arun-command


run: build stop
	mkdir -p ${CURDIR}/docker ${CURDIR}/artifacts/outputs
	cp -f ${CURDIR}/artifacts/build_darknet.sh ${CURDIR}/docker/build_darknet.sh

	docker run --rm -d --gpus all --name $(CONTAINER_NAME) \
		-v ${CURDIR}/docker:/workspace \
		-v ${CURDIR}/artifacts/outputs:/outputs \
		--env TRUE_USER=$(shell whoami) $(DOCKER_WINDOWS_ENV) $(DOCKER_UVA_ENV) \
		--env ACTUAL_PWD=$(ACTUAL_PWD) \
    	--env YOLO_TEMPLATES=$(YOLO_TEMPLATES) \
		--env BACKEND=darknet \
		darknet-container
	make logs


logs:
	@docker logs $(CONTAINER_NAME) --follow

collab: check-dataset-collab build
	@docker run --rm -it --gpus all --name $(CONTAINER_NAME) -v $(COLLAB_DIR):/workspace darknet-container

check-dataset-collab:
	cp $(COLLABFILES_DIR)/cars.cfg $(COLLAB_DIR)/cars.cfg;
	cp $(COLLABFILES_DIR)/cars.data $(COLLAB_DIR)/cars.data;
	cp $(COLLABFILES_DIR)/cars.names $(COLLAB_DIR)/cars.names;
	cp $(COLLABFILES_DIR)/run.py $(COLLAB_DIR)/run.py;
	cp $(COLLABFILES_DIR)/build_darknet.sh $(COLLAB_DIR)/build_darknet.sh;

build:
	@docker build \
		--build-arg UID=$(BUILD_UID) \
		--build-arg GID=$(BUILD_GID) \
		-t darknet-container \
		-f $(CURDIR)/Dockerfile $(PROJ_ROOT)

shell:
	@docker exec -it $(CONTAINER_NAME) /bin/bash

stop down:
	@-docker stop $(CONTAINER_NAME)
	@-docker rm $(CONTAINER_NAME)



# Build the apptainer image using the definition file
$(SIF_NAME): apptainer.def
	@$(MODULE_LOAD_APPTAINER); set -e; \
	cd "$(PROJ_ROOT)"; \
	apptainer build --force -f "$(SIF_NAME)" "$(CURDIR)/apptainer.def"


apptainer-build: $(SIF_NAME)

# Create the workspace directory if it doesn't exist
$(DARKNET_WORKSPACE):
	mkdir -p $(DARKNET_WORKSPACE)

delete:
	rm -rf ./docker
	rm -rf ./darknet_workspace
	-rm darknet_train.out darknet_train.err

# Run the apptainer container with a bind mount for a writable workspace
arun: apptainer-build $(DARKNET_WORKSPACE) arun-command
	
arun-command: $(DARKNET_WORKSPACE)
	mkdir -p $(DARKNET_WORKSPACE) \
	         ${CURDIR}/docker \
	         ${CURDIR}/artifacts/outputs
	cp -f ${CURDIR}/artifacts/build_darknet.sh ${CURDIR}/docker/build_darknet.sh

	@$(MODULE_LOAD_APPTAINER); \
	APPTAINERENV_UVA_VIRGINIA_RUNNING=$(UVA_VIRGINIA_RUNNING) \
	APPTAINERENV_ACTUAL_PWD=$(ACTUAL_PWD) \
	APPTAINERENV_YOLO_TEMPLATES=$(YOLO_TEMPLATES) \
	APPTAINERENV_DARKNET_PARENT=/host_workspace \
	apptainer run --nv --fakeroot \
	-B $(DARKNET_WORKSPACE):/host_workspace \
	-B ${CURDIR}/docker:/workspace \
	-B ${CURDIR}/artifacts/outputs:/outputs \
	$(SIF_NAME)


arun-cms-gpu:
	@echo "CUDA_VISIBLE_DEVICES is $$CUDA_VISIBLE_DEVICES"
	$(MODULE_LOAD_APPTAINER); apptainer exec --nv \
    	$(SIF_NAME) cms gpu watch --gpu=`echo $$CUDA_VISIBLE_DEVICES | tr -d '\n\r'` --delay=1 --dense --logfile=mylogfile.log &

arun-cars-command:
	@$(MODULE_LOAD_APPTAINER); SINGULARITYENV_UVA_VIRGINIA_RUNNING=$(UVA_VIRGINIA_RUNNING) SINGULARITYENV_ACTUAL_PWD=$(ACTUAL_PWD) \
		apptainer run --nv --fakeroot \
		-B $(DARKNET_WORKSPACE):/host_workspace \
		-B ${CURDIR}/CollabFiles:/workspace \
		$(SIF_NAME)

ashell:
	@$(MODULE_LOAD_APPTAINER); apptainer shell $(SIF_NAME)


slurm: delete apptainer-build
	@if echo "$$(dig -x $$(dig +short myip.opendns.com @resolver1.opendns.com) +short)" | grep -q "virginia.edu"; then \
	    ACCOUNT="--account=bi_dsc_community"; \
	else \
	    ACCOUNT=""; \
	fi; \
	echo "Submitting job with account option: $$ACCOUNT"; \
	sbatch -J "lego_train_$$(date +%Y%m%d_%H%M%S)" $$ACCOUNT legogears.slurm
	watch squeue -u $(USER)

slurm-shell:
	# srun -p gpu --gpus=v100:1 --account=bi_dsc_community --time=00:40:00 --mem=50gb --pty apptainer shell --nv $(SIF_NAME)
	srun -p gpu --gpus=geforce:1 --time=01:00:00 --mem=50gb --pty apptainer shell --nv $(SIF_NAME)


cars-slurm: delete apptainer-build $(DARKNET_WORKSPACE)
	sbatch -J "darknet_cars_train_$(date +%Y%m%d_%H%M%S)" CollabFiles/cars.slurm
