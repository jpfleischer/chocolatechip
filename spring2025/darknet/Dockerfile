# Use NVIDIA's CUDA base image
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install dependencies
RUN apt-get update
RUN apt-get install -y build-essential git libopencv-dev file python3-pip python3-venv



RUN apt install -y software-properties-common lsb-release && \
apt clean all

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null

RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 16FAAD7AF99A65E2

RUN apt update
RUN apt install cmake -y

RUN python3 -m venv /opt/venv
# Activate the virtual environment and upgrade pip
RUN . /opt/venv/bin/activate
RUN pip install --upgrade pip
RUN pip install cloudmesh-common
RUN pip install git+https://github.com/cloudmesh/cloudmesh-gpu.git

WORKDIR /workspace

# Build Darknet
CMD ["/bin/bash", "-c", "chmod +x script.sh; ./script.sh; cms gpu watch --gpu=`nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,` --delay=1 --dense --logfile=mylogfile.log & WATCH_PID=$!; python3 run.py; kill $WATCH_PID; target=$(ls -td /outputs/*/ | head -1); mv mylogfile.log \"$target\""]
# CMD ["/bin/bash", "-c", "chmod +x script.sh; ./script.sh; cms gpu watch --gpu=`nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,` --delay=1 --dense --logfile=mylogfile.log & python3 run.py"]
# CMD ["/bin/bash", "-c", "cms gpu watch --gpu=`nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,` --delay=1 --dense --logfile=mylogfile.log & sleep 25"]
