cmake_minimum_required(VERSION 3.28)

project(nvidia-devices LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 61)

enable_language(CUDA)

add_executable(nvidia-devices src/nvidia-devices.cu)
