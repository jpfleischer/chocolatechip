# Use NVIDIA's CUDA base image
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ARG APP_DIR=/opt/app
WORKDIR $APP_DIR

# System deps (rarely change)  ✅ stays cached
ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/usr/local/cuda/bin:/opt/venv/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Base deps (update+install together) — avoids stale apt indexes
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      build-essential git libopencv-dev file python3-venv python3-pip fio \
      ca-certificates wget gnupg lsb-release pkg-config; \
    rm -rf /var/lib/apt/lists/*

# Newer CMake from Kitware (no deprecated apt-key)
RUN set -eux; \
    wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc \
      | gpg --dearmor -o /etc/apt/trusted.gpg.d/kitware.gpg; \
    echo "deb [signed-by=/etc/apt/trusted.gpg.d/kitware.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/kitware.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends cmake; \
    rm -rf /var/lib/apt/lists/*

# Python venv (no activate needed in Docker)
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir \
      cloudmesh-common \
      git+https://github.com/cloudmesh/cloudmesh-gpu.git


COPY src/ ./src/
ENV PYTHONPATH=$APP_DIR/src

WORKDIR /workspace

# Build Darknet
# CMD ["sleep", "infinity"]

CMD ["bash", "-lc", "chmod +x script.sh; ./script.sh; python -u -m chocolatechip.model_training.train --profile LegoGearsDarknet"]

# CMD ["/bin/bash", "-c", "chmod +x script.sh; ./script.sh; cms gpu watch --gpu=`nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,` --delay=1 --dense --logfile=mylogfile.log & WATCH_PID=$!; python3 run.py; kill $WATCH_PID; target=$(ls -td /outputs/*/ | head -1); mv mylogfile.log \"$target\""]
# CMD ["/bin/bash", "-c", "chmod +x script.sh; ./script.sh; cms gpu watch --gpu=`nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,` --delay=1 --dense --logfile=mylogfile.log & python3 run.py"]
# CMD ["/bin/bash", "-c", "cms gpu watch --gpu=`nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,` --delay=1 --dense --logfile=mylogfile.log & sleep 25"]
