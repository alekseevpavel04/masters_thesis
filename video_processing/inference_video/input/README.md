# Place Input Videos Here

This directory is intended for storing input video files that will be processed by the application. The application will automatically detect and process all valid video files placed in this directory.

## Instructions

1. **Supported Formats**:
   - Ensure your video files are in one of the following formats:
     - `.mp4`
     - `.mkv`
     - `.avi`

2. **Placement**:
   - Copy or move your video files into this directory.
   - The application will process all valid video files found here.

3. **Processing**:
   - After placing the videos here, run the application.
   - The application will automatically detect and process all valid video files in this directory.

4. **Output**:
   - The processed (upscaled) videos will be saved in the `output` directory.

## Notes

- **Do not delete this file**: It serves as a placeholder to ensure the directory exists in your repository.
- **If the directory is empty**: The application will not process any videos. Make sure to place your input videos here before running the application.
- **File Naming**: Avoid using special characters or spaces in filenames to prevent issues during processing.

## Troubleshooting

- **No videos processed**:
  - Ensure that the video files are placed in this directory and are in a supported format.
  - Check the logs (in the `logs` directory) for any errors during processing.
- **Unsupported formats**:
  - If your video is in an unsupported format, convert it to `.mp4`, `.mkv`, or `.avi` using a video converter tool.