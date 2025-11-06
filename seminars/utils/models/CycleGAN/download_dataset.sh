#!/bin/bash

set -e  # Exit immediately on error

# Set dataset name
FILE="horse2zebra"

# Get the target directory from the command line, or default to the current directory
TARGET_DIR=${1:-$(pwd)}
FINAL_DIR="$TARGET_DIR/$FILE"  # The dataset will be extracted inside TARGET_DIR/horse2zebra

ZIP_FILE=~/Downloads/${FILE}-dataset.zip
KAGGLE_URL="https://www.kaggle.com/api/v1/datasets/download/balraj98/horse2zebra-dataset"

# Ensure the target directory exists
mkdir -p "$FINAL_DIR"

echo "Downloading dataset: ${FILE} from Kaggle..."
if ! curl -L -o "$ZIP_FILE" "$KAGGLE_URL"; then
    echo "Error: Failed to download ${FILE}.zip"
    exit 1
fi

echo "Extracting ${ZIP_FILE} into ${FINAL_DIR}..."
if ! unzip -q "$ZIP_FILE" -d "$FINAL_DIR"; then
    echo "Error: Failed to unzip ${ZIP_FILE}"
    rm -f "$ZIP_FILE"
    exit 1
fi
rm -f "$ZIP_FILE"

# Ensure train and test directories exist
mkdir -p "$FINAL_DIR/train" "$FINAL_DIR/test"

# Move datasets into the required structure
if [[ -d "$FINAL_DIR/trainA" && -d "$FINAL_DIR/trainB" && -d "$FINAL_DIR/testA" && -d "$FINAL_DIR/testB" ]]; then
    mv "$FINAL_DIR/trainA" "$FINAL_DIR/train/A"
    mv "$FINAL_DIR/trainB" "$FINAL_DIR/train/B"
    mv "$FINAL_DIR/testA" "$FINAL_DIR/test/A"
    mv "$FINAL_DIR/testB" "$FINAL_DIR/test/B"
    echo "Dataset successfully organized in: $FINAL_DIR"
else
    echo "Error: Expected dataset structure (trainA, trainB, testA, testB) not found!"
    exit 1
fi
