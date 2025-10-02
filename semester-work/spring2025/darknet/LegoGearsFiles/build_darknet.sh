#!/bin/bash

# Change to /host_workspace so that the darknet folder is created there
cd /host_workspace || { echo "Cannot change to /host_workspace"; exit 1; }

# Clone Darknet if not already present
if [ ! -d "darknet" ]; then
    git clone https://github.com/hank-ai/darknet.git
fi

# Change to the darknet directory
cd darknet || { echo "Cannot change to darknet directory"; exit 1; }

# Check if the "build" directory exists
if [ ! -d "build" ]; then
    mkdir build
    cd build || { echo "Cannot change to build directory"; exit 1; }
    
    # Check if nvidia-smi is available and working
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi &> /dev/null; then
        echo "nvidia-smi not found or failed; building Darknet in CPU-only mode."
        cmake .. -DDARKNET_TRY_CUDA=OFF -DDARKNET_TRY_ROCM=OFF
    else
        echo "nvidia-smi found; building Darknet with GPU support."
        cmake ..
    fi

    make -j$(nproc)
fi

# Attempt to install the package only if /var/lib/dpkg is writable
if [ -w /var/lib/dpkg ]; then
    dpkg -i /host_workspace/darknet/build/darknet-*.deb
else
    echo "Skipping package installation: /var/lib/dpkg is read-only."
fi
