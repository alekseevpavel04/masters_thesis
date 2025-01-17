# Inference Model Will Appear Here

This directory is where the pre-trained model files will be stored after running the `download_assets.sh` script. These files are essential for the video upscaling process.

## Instructions

1. **Download the Model**:
   - Run the `download_assets.sh` script to download the necessary model files:
     ```bash
     ./download_assets.sh
     ```
   - This script will automatically download the pre-trained model weights and place them in this directory.

2. **Expected Files**:
   - After running the script, the following files should appear in this directory:
     - `RealESRGAN_final.pth`: The pre-trained model weights for video upscaling.
     - (Optional) Other related assets, such as configuration files or TensorRT-optimized models.

3. **Usage**:
   - The application will automatically detect and use the model files from this directory during the upscaling process.
   - Ensure that the `RealESRGAN_final.pth` file is present before running the application.

## Notes

- **Do not delete this file**: It serves as a placeholder to ensure the directory exists in your repository.
- **If the directory is empty**: Make sure you have run the `download_assets.sh` script to download the model files.
- **Re-downloading**: If the model files are missing or corrupted, you can re-run the `download_assets.sh` script to download them again.

## Troubleshooting

- **Script not working**: Ensure that the script has execute permissions. You can set them using:
  ```bash
  chmod +x download_assets.sh