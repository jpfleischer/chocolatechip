# Use an older NVIDIA TensorRT container image.
FROM nvcr.io/nvidia/tensorrt:19.10-py3

# Set noninteractive mode.
ENV DEBIAN_FRONTEND=noninteractive

# Install additional system dependencies.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    protobuf-compiler \
    pybind11-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX version 1.4.1 (prebuilt wheels should be available with this image)
RUN pip3 install onnx==1.4.1

# Set the working directory.
WORKDIR /workspace

# Copy conversion files.
COPY yolo2onnx.py ./
COPY convert.sh ./
RUN chmod +x convert.sh

# Override the default entrypoint to bypass GPU checks.
ENTRYPOINT []

# Declare a volume for your model files.
VOLUME ["/data/traffic_training_cvat_unzipped/annotations"]

# Set the default command to run the conversion script.
CMD ["./convert.sh"]
