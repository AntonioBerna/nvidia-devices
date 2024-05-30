cmake_minimum_required(VERSION 3.28)

project(nvidia-devices LANGUAGES CXX CUDA)

enable_language(CUDA)

add_executable(nvidia-devices src/nvidia-devices.cu)
