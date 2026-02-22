#!/bin/bash

# Parent directory containing bag folders
BAGS_DIR="$1"

# Check if directory argument was provided
if [ -z "$BAGS_DIR" ]; then
  echo "Usage: $0 <bags_directory>"
  exit 1
fi

# Camera topic
IMAGE_TOPIC="/realsense_d456/color_image/compressed"
ODOM_TOPIC="/spot/odometry"

# Loop through each subdirectory
for BAG_PATH in "$BAGS_DIR"/*; do
  if [ -d "$BAG_PATH" ]; then
    BAG_NAME=$(basename "$BAG_PATH")
    OUTPUT_DIR="${BAG_PATH}/collected_images"

    echo "Processing bag: $BAG_NAME"
    echo "Output folder: $OUTPUT_DIR"

    mkdir -p "$OUTPUT_DIR"

    python3.10 ../examples/extract_images_from_rosbag.py "$BAG_PATH" "$IMAGE_TOPIC" "$ODOM_TOPIC" "$OUTPUT_DIR"

    echo "Finished: $BAG_NAME"
    echo "-----------------------------"
  fi
done

echo "All bags processed."

