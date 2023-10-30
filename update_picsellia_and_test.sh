#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <picsellia_version>"
    exit 1
fi

picsellia_version="$1"

folders=("yolov8-classification" "yolov8-segmentation-refactored" "yolov8-detection-refactored" "unet-instance-segmentation")

for script_dir in "${folders[@]}"; do
  if [ -f "$script_dir/requirements.txt" ]; then
    current_version=$(grep -o 'picsellia==[^ ]*' "$script_dir/requirements.txt")
    current_version=${current_version#*==}

    if [ "$current_version" = "$picsellia_version" ]; then
      echo "Version $picsellia_version is already in $script_dir/requirements.txt"
      exit 1
    else
      echo "Updating picsellia version in $script_dir's requirements file from $current_version to $picsellia_version.."
      # Update the picsellia version in the requirements file
      sed -i "s/picsellia==.*/picsellia==$picsellia_version/" "$script_dir/requirements.txt"
    fi
  else
    echo "Requirements file not found in $script_dir"
  fi
done

./run_unit_tests.sh all
./rebuild_test.sh


