PROJ_ROOT 	   := $(shell git -C "$(CURDIR)" rev-parse --show-toplevel)
SOURCE_PATH    := semester-work/fall2025/ultralytics
SIF_NAME	   := $(CURDIR)/my_ultralytics_container.sif
APPTAINER_SRC := $(PROJ_ROOT)/src
DATASET_DIR ?= $(CURDIR)/LegoGears_v2
ULTRA_WORKSPACE := $(CURDIR)/ultra_workspace

# Detect Git Bash vs Linux for a tiny path tweak (optional but nice)
UNAME_S := $(shell uname -s)
IS_WINDOWS := $(or $(findstring MINGW,$(UNAME_S)),$(findstring MSYS,$(UNAME_S)))
ifeq ($(IS_WINDOWS),)
  CUR_HOST := $(CURDIR)
  SHELL_ENV :=
  BUILD_UID := $$(id -u)
  BUILD_GID := $$(id -g)
else
  # Convert current dir to Windows-style path with forward slashes (C:/...)
  CUR_HOST := $(shell cygpath -m "$(CURDIR)")
  # Prevent path mangling on -v args
  SHELL_ENV := MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL=*
  BUILD_UID := 1000
  BUILD_GID := 1000
endif

benchmark-ultra benchmarking benchmark:
	@mkdir -p outputs
	docker build \
		--build-arg UID=$(BUILD_UID) \
		--build-arg GID=$(BUILD_GID) \
		-t my-benchmark-app \
		-f "$(CURDIR)/Dockerfile" "$(PROJ_ROOT)"
	$(SHELL_ENV) docker run --rm -it --shm-size=4gb --gpus all \
		-e TRUE_USER=$$(id -un) \
		-e HOME=/home/appuser \
		-e ACTUAL_PWD="$(CUR_HOST)" \
		-v "$(CUR_HOST)/LegoGears_v2":/workspace/LegoGears_v2 \
		-v "$(CUR_HOST)/outputs":/outputs \
		my-benchmark-app

ultra bash:
	@mkdir -p outputs
	$(SHELL_ENV) docker run -it --shm-size=4gb --gpus all \
		-v "$(CUR_HOST)/LegoGears_v2":/workspace/LegoGears_v2 \
		-v "$(CUR_HOST)/outputs":/outputs \
		my-benchmark-app \
		/bin/bash -lc "ls -la /ultralytics && exec /bin/bash"

# on hipergator
UVA_VIRGINIA_RUNNING := $(shell \
	if echo "$$(dig -x $$(dig +short myip.opendns.com @resolver1.opendns.com) +short)" | grep -q "virginia.edu"; then \
		echo true; \
	else \
		echo false; \
	fi)

DOCKER_UVA_ENV := --env UVA_VIRGINIA_RUNNING=$(UVA_VIRGINIA_RUNNING)

MODULE_LOAD_APPTAINER = if [ "$(UVA_VIRGINIA_RUNNING)" = "true" ]; then \
		. /etc/profile.d/modules.sh && module load apptainer; \
	fi

$(SIF_NAME): apptainer.def
	@$(MODULE_LOAD_APPTAINER); set -e; \
	cd "$(PROJ_ROOT)"; \
	APPTAINER_SRC="$(APPTAINER_SRC)" apptainer build --force -f "$(SIF_NAME)" "$(CURDIR)/apptainer.def"

$(ULTRA_WORKSPACE):
	mkdir -p $(ULTRA_WORKSPACE)/.home \
	         $(ULTRA_WORKSPACE)/.cache/mpl \
	         $(CURDIR)/artifacts/outputs

apptainer-build: $(SIF_NAME)

slurm: apptainer-build
	@if echo "$$(dig -x $$(dig +short myip.opendns.com @resolver1.opendns.com) +short)" | grep -q "virginia.edu"; then \
	    ACCOUNT="--account=bi_dsc_community"; \
	else \
	    ACCOUNT=""; \
	fi; \
	echo "Submitting job with account option: $$ACCOUNT"; \
	sbatch -J "lego_train_$$(date +%Y%m%d_%H%M%S)" $$ACCOUNT legogears.slurm
	watch squeue -u $(USER)

apptainer-run: $(ULTRA_WORKSPACE)
	APPTAINERENV_HOME=/workspace/.home \
	APPTAINERENV_XDG_CACHE_HOME=/workspace/.cache \
	APPTAINERENV_XDG_CONFIG_HOME=/workspace/.config \
	APPTAINERENV_MPLCONFIGDIR=/workspace/.cache/mpl \
	APPTAINERENV_DATA_ROOT=/workspace \
	APPTAINERENV_PROJECT_ROOT=/workspace/project \
	# optional if your trainer reads it
	APPTAINERENV_ULTRA_WORKERS=1 \
	apptainer run --nv --fakeroot \
		--contain --no-home --home /workspace/.home \
		-B "$(ULTRA_WORKSPACE)":/workspace \
		-B "$(CURDIR)":/workspace/project \
		-B "$(CURDIR)/artifacts/outputs":/outputs \
		my_ultralytics_container.sif



# yolo detect train data=LG_v2.yaml model=yolo11n.pt epochs=2134 batch=64
# yolo detect train data=LG_v2.yaml model=yolo11n.pt epochs=1 batch=64

#make apptainer.def and whatever is in it
#make legogears.slurm