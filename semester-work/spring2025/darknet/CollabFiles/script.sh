#!/bin/bash

if [ ! -d "darknet" ]; then
    git clone https://github.com/hank-ai/darknet.git
fi

# Set working directory
cd /workspace/darknet

# Check if the "build" directory exists
if [ ! -d "build" ]; then
    # Create and enter the build directory
    mkdir build
    cd build

    # Build Darknet
    cmake ..
    make -j$(nproc) package
fi 

# Install the package
dpkg -i /workspace/darknet/build/darknet-*.deb

mv /workspace/cars.cfg /workspace/unzips
mv /workspace/cars.data /workspace/unzips
mv /workspace/cars.names /workspace/unzips
