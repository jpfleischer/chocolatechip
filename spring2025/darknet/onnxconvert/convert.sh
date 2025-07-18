#!/bin/bash
set -e

#MODEL_NAME="combined_best"
#MODEL_DIR="/data/ke_images_newrepo/darknet/data/combined"
MODEL_NAME="yash_images_best"
MODEL_DIR="/data/yash_images/"

mkdir -p output

echo "Copying model files from ${MODEL_DIR}..."
cp ${MODEL_DIR}/${MODEL_NAME}.weights .
cp ${MODEL_DIR}/*.cfg ${MODEL_NAME}.cfg

echo "Converting DarkNet model to ONNX..."
python3 yolo2onnx.py -c ${MODEL_NAME}.cfg -w ${MODEL_NAME}.weights -o output

echo "Conversion complete. ONNX file generated in the output directory."

# mv output/*onnx /workspace/output

echo "Listing output directory contents:"
ls -l output
