# TorchCor Docker Support

This folder contains a simple Docker setup for running EP simulation with TorchCor on a left atrium, with GPU support using NVIDIA GPUs. 

## Requirements

- Ubuntu or Linux host
- NVIDIA GPU with compatible driver (tested with NVIDIA T400, Driver Version ≥ 535, CUDA 12.2)
- Docker (≥ 20.10)
- NVIDIA Container Toolkit

---

## 1. Install / Configure NVIDIA Container Toolkit

This allows Docker containers to access your GPU:

```bash
# Add the NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install the NVIDIA Docker toolkit
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker to apply changes
sudo systemctl restart docker
```

## 2. Verify GPU access in Docker

To make sure Docker can see your GPU:
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

You should see output with your GPU information.

## 3. Build the docker image

First, decompress the Case_10.zip in the current folder. In the docker folder, run
```bash
docker build -t torchcor .
```
This will build the docker image


## 3. Run the docker image
```bash
docker run --rm -v ./:/app torchcor
```
You will see the log information in the terminal, and when the program finishes the output files will be located in side the folder named `atrium`. 