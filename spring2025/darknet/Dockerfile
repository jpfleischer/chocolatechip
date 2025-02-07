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

# Manually install latest CMake
WORKDIR /tmp
RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.6/cmake-3.27.6-linux-x86_64.tar.gz && \
    tar -xzvf cmake-3.27.6-linux-x86_64.tar.gz && \
    mv cmake-3.27.6-linux-x86_64 /usr/local/cmake && \
    ln -s /usr/local/cmake/bin/* /usr/local/bin/ && \
    rm cmake-3.27.6-linux-x86_64.tar.gz

WORKDIR /workspace
# Build Darknet
# COPY script.sh /workspace
# # set up lego gears training
# COPY train_setup.py /workspace
# COPY LegoGears_v2 /workspace/LegoGears_v2
# COPY LegoGears.cfg /workspace/LegoGears_v2/LegoGears.cfg
# CMD ["/bin/bash", "-c", "exec /bin/bash"]
CMD ["/bin/bash", "-c", "chmod +x script.sh && ./script.sh && darknet version && python3 benchmark.py && exec /bin/bash"]