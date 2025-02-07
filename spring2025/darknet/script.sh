#!/bin/bash
# Build Darknet
cd /workspace/darknet
mkdir build
cd build
cmake ..
make -j$(nproc) package
dpkg -i build/darknet-*.deb
darknet version