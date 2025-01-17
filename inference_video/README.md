# Video Upscaling Inference Module

This module is part of a larger project focused on AI-based video upscaling. It is responsible for the inference (processing) of video files using the **RRDBNet** model, with optional TensorRT optimization for improved performance on NVIDIA GPUs.

---

## Project Structure

This module is not a standalone application and should be used as part of the larger project. Below is the structure of this module:

- `input/`: Place your input video files here. See [input/README.md](input/README.md).
- `output/`: Processed (upscaled) videos will appear here. See [output/README.md](output/README.md).
- `model/`: Pre-trained model files will appear here after running `download_assets.sh`. See [model/README.md](model/README.md).
- `logs/`: Log files will appear here. See [logs/README.md](logs/README.md).



---

## Installation

This module is part of a larger project and should not be installed separately. To set up the entire project, follow the installation instructions in the root `README.md` of the main repository.

---

## Usage

1. **Place your input videos**:
   - Copy your video files into the `input` directory.

2. **Run the application**:
   - Execute the following command to start the inference process:
     ```
     python inference.py
     ```

3. **Check the output**:
   - The upscaled videos will be saved in the `output` directory.

## Performance

Below are the performance metrics for different input resolutions using a **GeForce RTX 3070** GPU:

- **Input: 480x360 → Output: 960x720**:
  - Average FPS: **12 FPS**

- **Input: 640x480 → Output: 1280x960**:
  - Average FPS: **5 FPS**

## Installation Guide for cuDNN, TensorRT and Torch2TRT

### 1. Make sure CUDA is installed

Ensure that CUDA is installed and properly configured on your system. You can verify this by running:
```
nvcc --version
```

### 2. Install cuDNN

1. Install cuDNN:
```
sudo apt-get -y install cudnn-cuda-12
```

2. Verify installation:
```
dpkg -l | grep cudnn
```

### 3. Install TensorRT

Follow these steps to install TensorRT:

1. Install TensorRT using pip:
```
python3 -m pip install tensorrt-cu12 tensorrt-lean-cu12 tensorrt-dispatch-cu12
```

2. Alternatively, install TensorRT using apt:
```
sudo apt-get install tensorrt
```

3. Verify installation:
```
dpkg -l | grep nvinfer
```

### 4. Install torch2trt

Follow these steps to install torch2trt:

1. Clone the torch2trt repository:
```
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
```

2. Navigate to the cloned repository:
```
cd torch2trt
```

3. Install torch2trt:
```
sudo path_to_your_Python_environment setup.py install --plugins
```

4. Alternatively, follow the installation guide here: [TensorRT Torch2TRT Installation Guide](https://github.com/vujadeyoon/TensorRT-Torch2TRT)

---

## Troubleshooting

- **No videos processed**: Ensure that the input videos are placed in the `input` directory and are in a supported format.
- **Model not found**: Verify that the model files are downloaded and placed in the `model` directory.
- **CUDA/cuDNN issues**: Ensure that CUDA and cuDNN are properly installed and configured.

