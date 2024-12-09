# NVIDIA Devices

## What is this software for?

This is a project for educational purposes only to introduce beginners to parallel programming on GPUs. In fact, in this project a very simple `CUDA C++` code is implemented that is capable of recognizing how many and which GPUs are connected to our computer and furthermore for each of them we print the main information that characterizes them, for example the maximum number of threads per multiprocessor, the maximum size of a block, etc...

> [!IMPORTANT]
> This repository was created with the aim of helping those who can't understand how to get started in the world of GPUs and in particular parallel programming on GPUs. So if you find something unclear and that you want to explore further, open an issue in the dedicated section.

## How to use CUDA C++ in Linux distros

The first step is to download the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). Thanks to it we will also have the `nvcc` compiler, fundamental for this project. I also recommend you download the `nvidia-settings` package which will allow you to manage all the information regarding the GPU or GPUs via GUI.

> [!TIP]
> If you use the `Manjaro Linux` distribution you are at an advantage, in fact to install the NVIDIA drivers necessary for the correct recognition of the GPU and for the correct use of it via `CUDA C++`, you will simply have to go to `Manjaro Settings Manager > Hardware Configuration` and then you will simply have to install the latest (i.e. top-of-the-line) drivers for your GPU. Surely on other Linux distributions there will be other ways, perhaps similar to this, but which I don't know for now.

Once the installation of all the tools I mentioned previously (including the drivers) has been completed, run this command:

```
nvcc --version
```

and you should get the version, i.e. something like this:

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Wed_Apr_17_19:19:55_PDT_2024
Cuda compilation tools, release 12.5, V12.5.40
Build cuda_12.5.r12.5/compiler.34177558_0
```

and also using the command:

```
nvidia-smi
```

if you have installed everything correctly you should get something similar to this:

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.78                 Driver Version: 550.78         CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce MX150           Off |   00000000:01:00.0 Off |                  N/A |
| N/A   46C    P8             N/A / ERR!  |       5MiB /   2048MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      1297      G   /usr/lib/Xorg                                   4MiB |
+-----------------------------------------------------------------------------------------+
```

> [!NOTE]
> The `nvidia-smi` (NVIDIA System Management Interface) command is a command line tool provided by NVIDIA to monitor and manage NVIDIA GPUs. It is mainly used to get detailed information about the GPU hardware and its usage, as well as allowing some management operations.

Finally these are the hardware and software specifications regarding my setup:

| Operating System     | GPU                  | CUDA  |
| :---:                | :---:                | :---: |
| Manjaro Linux 24.0.1 | NVIDIA GeForce MX150 | v12.5 |

## Mini docs

At this point the bulk of the work has been done. In fact, it will be sufficient to download this repository with the following command:

```
git clone https://github.com/AntonioBerna/nvidia-devices.git
```

by entering the `nvidia-devices` working directory you will find the `Makefile` file, then using the following command:

```
make
```

and finally:

```
./bin/nvidia-devices
```

we obtain:

```
===== CUDA DEVICE 0 COMPREHENSIVE REPORT =====
Device Name: NVIDIA GeForce MX150
Architecture: Pascal (Compute 6.1)
Total Global Memory: 1.95 GB

--- Warp Characteristics ---
Warp Size: 32 threads
Max Active Warps per SM: 64

--- Advanced Capabilities ---
Compute Capabilities:
 - Async Engine Count: 2
 - Unified Addressing: Supported
 - Managed Memory: Supported
 - Concurrent Kernel Execution: Supported
 - Cooperative Launch: Supported

--- Advanced Memory Specifications ---
Memory Types:
 - Surface Memory: Supported
 - Texture Memory: Supported

Cache Specifications:
 - L2 Cache Size: 512 KB

--- Numerical Compute Capabilities ---
Double Precision Performance: Supported
Tensor Cores: Not Available
```
