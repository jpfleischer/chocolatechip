#!/bin/bash
set -e

# Define your model name (without extension).
MODEL_NAME="annotations_final"

# Path where your custom model files (both .weights and .cfg) are located.
MODEL_DIR="/data/traffic_training_cvat_unzipped/annotations"

# Create an output directory if it doesn't exist.
mkdir -p output

echo "Copying model files from ${MODEL_DIR}..."
cp ${MODEL_DIR}/${MODEL_NAME}.weights .
cp ${MODEL_DIR}/annotations.cfg ${MODEL_NAME}.cfg

echo "Converting DarkNet model to ONNX..."
python3 yolo_to_onnx.py -m ${MODEL_NAME}

echo "Converting ONNX model to TensorRT engine..."
python3 onnx_to_tensorrt.py -m ${MODEL_NAME}

echo "Moving generated files to output directory..."
mv ${MODEL_NAME}.trt output/
mv ${MODEL_NAME}.onnx output/
mv ${MODEL_NAME}.cfg output/

echo "Conversion complete. ONNX and TensorRT engine files generated in the output directory."
