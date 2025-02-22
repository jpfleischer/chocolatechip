# Use NVIDIA's CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install dependencies including NVIDIA tools
RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get install -y git
RUN apt-get install -y libopencv-dev 
RUN apt-get install -y wget
RUN apt-get install -y file
RUN apt-get install -y python3-pip
RUN pip install cloudmesh-common
RUN pip install cloudmesh-gpu

# Manually install latest CMake
WORKDIR /tmp
RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.6/cmake-3.27.6-linux-x86_64.tar.gz && \
    tar -xzvf cmake-3.27.6-linux-x86_64.tar.gz && \
    mv cmake-3.27.6-linux-x86_64 /usr/local/cmake && \
    ln -s /usr/local/cmake/bin/* /usr/local/bin/ && \
    rm cmake-3.27.6-linux-x86_64.tar.gz

WORKDIR /workspace

# Build Darknet
CMD ["/bin/bash", "-c", "chmod +x script.sh && ./script.sh && darknet version && python3 run.py && exec /bin/bash"]