#!/bin/bash

# Set working directory
cd /workspace/darknet

# Check if the "build" directory exists
if [ -d "build" ]; then
    echo "Darknet is already built. Exiting."
    exit 0
fi

# Create and enter the build directory
mkdir build
cd build

# Build Darknet
cmake ..
make -j$(nproc) package

# Install the package
dpkg -i /workspace/darknet/build/darknet-*.deb