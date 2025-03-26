# Use NVIDIA's CUDA base image
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install dependencies
RUN apt-get update
RUN apt-get install -y build-essential git libopencv-dev file cmake python3-pip 
RUN pip install --break-system-packages cloudmesh-common
RUN pip install --break-system-packages git+https://github.com/cloudmesh/cloudmesh-gpu.git

WORKDIR /workspace

# Build Darknet
CMD ["/bin/bash", "-c", "chmod +x script.sh && ./script.sh && cms gpu watch --gpu=$(echo $CUDA_VISIBLE_DEVICES | tr -d '\\n\\r') --delay=1 --dense --logfile=mylogfile.log & darknet version && python3 run.py && exec /bin/bash"]
