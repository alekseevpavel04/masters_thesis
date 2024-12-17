#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Define the target directory relative to the script's location
TARGET_DIR="$SCRIPT_DIR/assets/weights/"

# Create the target directory if it does not exist
if [ ! -d "$TARGET_DIR" ]; then
    echo "Creating target directory: $TARGET_DIR"
    mkdir -p "$TARGET_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create target directory $TARGET_DIR."
        exit 1
    fi
else
    echo "Directory already exists: $TARGET_DIR"
fi

# Print the actual directory where files will be saved
echo "Target directory for downloads: $TARGET_DIR"

echo "Downloading weights..."

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
FILE1_URL="https://drive.google.com/uc?id=1QY0wB9VmrCLOyn3HcL7dhOkRH8uBOuSZ"
FILE2_URL="https://drive.google.com/uc?id=1IA-oHhhxEil2luFEsoFxsSH6jXMQwyeH"

# Download files
download_file "$FILE1_URL" "$TARGET_DIR/RealESRGAN_x2plus_netD.pth"
download_file "$FILE2_URL" "$TARGET_DIR/RealESRGAN_x2plus.pth"

echo "Download complete. Files saved to: $TARGET_DIR"
