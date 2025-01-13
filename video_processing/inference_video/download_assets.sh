#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Define the target directory relative to the script's location
TARGET_DIR_final="$SCRIPT_DIR/model/"

# Create the target directory if it does not exist
if [ ! -d "$TARGET_DIR_final" ]; then
    echo "Creating target directory: $TARGET_DIR_final"
    mkdir -p "$TARGET_DIR_final"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create target directory $TARGET_DIR_final."
        exit 1
    fi
else
    echo "Directory already exists: $TARGET_DIR_final"
fi

# Define the directory for the video example
VIDEO_DIR="$SCRIPT_DIR/input/"

# Create the video example directory if it does not exist
if [ ! -d "$VIDEO_DIR" ]; then
    echo "Creating video example directory: $VIDEO_DIR"
    mkdir -p "$VIDEO_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create video example directory $VIDEO_DIR."
        exit 1
    fi
else
    echo "Directory already exists: $VIDEO_DIR"
fi

# Print the actual directories where files will be saved
echo "Target directory for final model: $TARGET_DIR_final"
echo "Target directory for video example: $VIDEO_DIR"
echo "Downloading files..."

# Function to download files and check for errors
download_file() {
    local file_url=$1
    local destination=$2

    gdown -O "$destination" "$file_url"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download file from $file_url."
        exit 1
    else
        echo "File downloaded successfully: $destination"
    fi
}

# File URLs to download
FILE_final_URL="https://drive.google.com/uc?id=1IA-oHhhxEil2luFEsoFxsSH6jXMQwyeH"
VIDEO_EXAMPLE_URL="https://drive.google.com/uc?id=1bPbxJvrbETkf0P53JFDuRQ11YtM4HXwD"

# Download files
download_file "$FILE_final_URL" "$TARGET_DIR_final/RealESRGAN_final.pth"
download_file "$VIDEO_EXAMPLE_URL" "$VIDEO_DIR/video_example.mkv"

echo "Download complete. Files saved to: $TARGET_DIR_final and $VIDEO_DIR"
