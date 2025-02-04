# Use NVIDIA's CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install dependencies including NVIDIA tools
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libopencv-dev \
    wget \
    nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc \
    && rm -rf /var/lib/apt/lists/*

# Manually install latest CMake
WORKDIR /tmp
RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.6/cmake-3.27.6-linux-x86_64.tar.gz && \
    tar -xzvf cmake-3.27.6-linux-x86_64.tar.gz && \
    mv cmake-3.27.6-linux-x86_64 /usr/local/cmake && \
    ln -s /usr/local/cmake/bin/* /usr/local/bin/ && \
    rm cmake-3.27.6-linux-x86_64.tar.gz

# Clone Hank AI Darknet
WORKDIR /workspace
RUN git clone https://github.com/hank-ai/darknet.git

# Build Darknet
WORKDIR /workspace/darknet
RUN mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) package

# Install the built package
RUN dpkg -i build/darknet-*.deb

# Verify installation
RUN darknet version

WORKDIR /workspace

CMD ["/bin/bash"]

