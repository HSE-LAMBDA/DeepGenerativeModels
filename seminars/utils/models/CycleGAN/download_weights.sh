#!/bin/bash

# Script to download weights for specific datasets for CycleGAN

# Input arguments
FILE=$1
OUTPUT_DIR=${2:-.}  # Default to the current directory if no output directory is specified

# List of valid datasets
VALID_DATASETS=(
  "ae_photos" "apple2orange" "summer2winter_yosemite"
  "horse2zebra" "monet2photo" "cezanne2photo"
  "ukiyoe2photo" "vangogh2photo" "maps"
  "cityscapes" "facades" "iphone2dslr_flower"
)

# Function to check if the dataset is valid
is_valid_dataset() {
  for dataset in "${VALID_DATASETS[@]}"; do
    if [[ "$dataset" == "$1" ]]; then
      return 0
    fi
  done
  return 1
}

# Validate input
if ! is_valid_dataset "${FILE}"; then
  echo "Available datasets are: ${VALID_DATASETS[*]}"
  exit 1
fi

# Set URL and file names
URL="https://github.com/Lornatang/CycleGAN-PyTorch/releases/download/1.0/${FILE}.zip"
ZIP_FILE="${FILE}.zip"

# Download dataset
if command -v wget &> /dev/null; then
  wget -N "${URL}" -O "${ZIP_FILE}"
elif command -v curl &> /dev/null; then
  curl -L "${URL}" -o "${ZIP_FILE}"
else
  echo "Error: Neither wget nor curl is installed."
  exit 1
fi

# Ensure the output directory exists
mkdir -p "${OUTPUT_DIR}"

# Unzip to a temporary location and move to the output directory
unzip -o "${ZIP_FILE}" -d . && rm "${ZIP_FILE}"
mv "${FILE}" "${OUTPUT_DIR}/"

echo "Weights downloaded and extracted to: ${OUTPUT_DIR}/${FILE}"