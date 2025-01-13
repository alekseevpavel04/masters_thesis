# Installation Guide for TensorRT and Torch2TRT

## 1. Install CUDA

## 2. Install cuDNN

1. Install cuDNN:
   ```bash
   sudo apt-get -y install cudnn-cuda-12
   ```
2. Verify installation:
   ```bash
   dpkg -l | grep cudnn
   ```

## 3. Install TensorRT

Follow these steps to install TensorRT:

1. Install TensorRT using pip:
   ```bash
   python3 -m pip install tensorrt-cu12 tensorrt-lean-cu12 tensorrt-dispatch-cu12
   ```
2. Alternatively, install TensorRT using apt:
   ```bash
   sudo apt-get install tensorrt
   ```
3. Verify installation:
   ```bash
   dpkg -l | grep nvinfer
   ```

## 4. Install torch2trt

Follow these steps to install torch2trt:

1. Clone the torch2trt repository:
   ```bash
   git clone https://github.com/NVIDIA-AI-IOT/torch2trt
   ```
2. Navigate to the cloned repository:
   ```bash
   cd torch2trt
   ```
3. Install torch2trt:
   ```bash
   sudo /mnt/d/wsl_projects_main_root/master_thesis/masters_thesis/venv/bin/python setup.py install --plugins
   ```
   *(Replace `/mnt/d/wsl_projects_main_root/master_thesis/masters_thesis/venv/bin/python` with the path to your Python environment as shown by `which python`)*
4. Alternatively, follow the installation guide here: [TensorRT Torch2TRT Installation Guide](https://github.com/vujadeyoon/TensorRT-Torch2TRT)

   
- Basic speed is 1.16 FPS on average for 960x720 input
