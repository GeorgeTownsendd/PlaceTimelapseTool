#!/bin/bash

# Get the script's own directory
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Change to the script's directory
cd "$dir"

# Find the smallest numbered file
start=$(ls frame*.jpg | sort -V | head -1 | grep -o '[0-9]\+')

# Run the ffmpeg command with the correct start number
ffmpeg -y -framerate 30 -start_number $start -i frame%d.jpg -c:v libx264 -pix_fmt yuv420p output.mp4
