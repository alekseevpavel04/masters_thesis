# Dataset Preparation Module

This module is responsible for creating the dataset by extracting frames from raw videos, filtering out low-quality images, and removing duplicates.

---

## Project Structure

This module is not a standalone application and should be used as part of the larger project. Below is the structure of this module:

- `raw_video/`: Place your raw video files here. See [README.md](raw_video/README.md).
- `output_images/`: Extracted frames will appear here. See [README.md](output_images/README.md).
- `logs/`: Log files will appear here. See [README.md](logs/README.md).
---

## Installation

This module is part of a larger project and should not be installed separately. To set up the entire project, follow the installation instructions in the root `README.md` of the main repository.

---

## Usage

1. **Place your raw videos**:
   - Copy your video files into the `raw_video` directory.

2. **Run the dataset preparation script**:
   - Execute the following command to start the dataset creation process:
     ```
     python preprocess_data.py
     ```

3. **Check the output**:
   - The extracted frames will be saved in the `output_images` directory.
   - Logs will be saved in the `logs` directory.

---

## Notes

- **Do not delete placeholder files**: They ensure the directories exist in your repository.
- **If directories are empty**: Make sure to place your raw videos in the `raw_video` directory before running the script.