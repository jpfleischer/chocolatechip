# Dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ARG APP_DIR=/opt/app
WORKDIR $APP_DIR

# --- Create user that can match the host IDs (supplied at build time) ---
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appgroup \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash appuser

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/usr/local/cuda/bin:/opt/venv/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Base deps
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      build-essential git libopencv-dev file python3-venv python3-pip fio \
      ca-certificates wget gnupg lsb-release pkg-config; \
    rm -rf /var/lib/apt/lists/*

# Newer CMake
RUN set -eux; \
    wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc \
      | gpg --dearmor -o /etc/apt/trusted.gpg.d/kitware.gpg; \
    echo "deb [signed-by=/etc/apt/trusted.gpg.d/kitware.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/kitware.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends cmake; \
    rm -rf /var/lib/apt/lists/*

# Python venv + deps
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir \
      pycocotools \
      imagesize \
      cloudmesh-common \
      git+https://github.com/cloudmesh/cloudmesh-gpu.git

# Your Python package
COPY --chown=appuser:appgroup src/ ./src/
ENV PYTHONPATH=$APP_DIR/src

# Default in-image paths (binds will overlay these)
RUN mkdir -p /outputs /workspace \
 && chown -R appuser:appgroup /outputs /workspace

# Optional: collaborative-friendly umask
RUN echo 'umask 0002' >> /etc/profile
ENV HOME=/home/appuser
USER appuser
WORKDIR /workspace

# Run the script via bash (don’t rely on chmod on a bind mount)
CMD ["bash", "-lc", "\
bash build_darknet.sh; \
python -u -m chocolatechip.model_training.train --profile LegoGearsDarknet"]

# CMD ["bash", "-lc", "chmod +x build_darknet.sh; ./build_darknet.sh; python -u -m chocolatechip.model_training.train --profile LegoGearsDarknet"]
# CMD ["/bin/bash", "-c", "chmod +x build_darknet.sh; ./build_darknet.sh; cms gpu watch --gpu=`nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,` --delay=1 --dense --logfile=mylogfile.log & WATCH_PID=$!; python3 run.py; kill $WATCH_PID; target=$(ls -td /outputs/*/ | head -1); mv mylogfile.log \"$target\""]
# CMD ["/bin/bash", "-c", "chmod +x build_darknet.sh; ./build_darknet.sh; cms gpu watch --gpu=`nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,` --delay=1 --dense --logfile=mylogfile.log & python3 run.py"]
# CMD ["/bin/bash", "-c", "cms gpu watch --gpu=`nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,` --delay=1 --dense --logfile=mylogfile.log & sleep 25"]
