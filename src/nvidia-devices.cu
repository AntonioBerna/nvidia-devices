#include <iostream>
#include <cuda_runtime.h>

const float BYTES_PER_MEGABYTE = 1048576.0f;

int main(void) {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        std::cout << "cudaGetDeviceCount returned " << static_cast<int>(error_id) << "\n" << cudaGetErrorString(error_id) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "There are no available device(s) that support CUDA" << std::endl;
    } else {
        std::cout << "Detected " << deviceCount << " CUDA Capable device(s)" << std::endl;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << "\nDevice " << device << ": \"" << deviceProp.name << "\"" << std::endl;
        std::cout << "\tTotal amount of global memory: " << deviceProp.totalGlobalMem / BYTES_PER_MEGABYTE << " MB" << std::endl;
        std::cout << "\tNumber of multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "\tMax threads per multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "\tMax threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "\tMax block dimensions: [" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << "]" << std::endl;
        std::cout << "\tMax grid dimensions: [" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << "]" << std::endl;
        std::cout << "\tCompute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    }

    return 0;
}

