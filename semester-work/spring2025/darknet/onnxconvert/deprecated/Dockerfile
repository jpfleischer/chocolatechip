# Choose a suitable tag from https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt
# e.g. 23.08-py3, 23.06-py3, 22.12-py3, etc.
FROM nvcr.io/nvidia/tensorrt:23.08-py3

# Non-interactive apt config
ENV DEBIAN_FRONTEND=noninteractive

# Optional: install any additional packages you need.
# TensorRT dev libraries, Python, CUDA are already present in this container.
RUN apt-get update && apt-get install -y \
    git \
    libopencv-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# If you need specific Python packages (beyond what's already in the TRT image):
RUN pip3 install onnx onnxruntime opencv-python

WORKDIR /workspace

# Clone the tensorrt_demos repo
RUN git clone https://github.com/jkjung-avt/tensorrt_demos.git

# Patch gpu_cc.py to set the GPU architecture.
RUN sed -i '/^def get_gpu_archs():/,/^def /d' tensorrt_demos/plugins/gpu_cc.py && \
    printf "def get_gpu_archs():\n    return ['86']\n" >> tensorrt_demos/plugins/gpu_cc.py

# Adjust the Makefile if needed. For example, you may need to:
# 1) Update the TensorRT include path if tensorrt_demos is pointing
#    to /usr/local/TensorRT-7.x/ but the container uses /usr/include/x86_64-linux-gnu.
# 2) Update library paths similarly.

# For example, if the Makefile references /usr/local/TensorRT-7.1.3.4/include:
RUN sed -i "s#/usr/local/TensorRT-7.1.3.4/include#/usr/include/x86_64-linux-gnu#g" \
    tensorrt_demos/plugins/Makefile
# And similarly for the TensorRT lib path if needed:
RUN sed -i "s#/usr/local/TensorRT-7.1.3.4/lib#/usr/lib/x86_64-linux-gnu#g" \
    tensorrt_demos/plugins/Makefile

# Build the plugins
WORKDIR /workspace/tensorrt_demos/plugins
RUN make

# Switch to yolo folder. (Optional depending on your usage.)
WORKDIR /workspace/tensorrt_demos/yolo

# Copy your own conversion script in if desired
COPY convert.sh convert.sh
RUN chmod +x convert.sh

# (Optional) Provide a volume mount
VOLUME ["/data"]

# Default command to run your script
CMD ["./convert.sh"]
