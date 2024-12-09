#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cuda_runtime.h>

const float BYTES_PER_GIGABYTE = 1073741824.0f;

class CUDADeviceInspector {
private:
    std::string getArchitectureName(int major, int minor) {
        static const struct {
            int major, minor;
            const char *name;
        } architectures[] = {
            {1, 0, "Tesla"},
            {1, 1, "Tesla"},
            {1, 2, "Tesla"},
            {1, 3, "Fermi"},
            {2, 0, "Fermi"},
            {2, 1, "Fermi"},
            {3, 0, "Kepler"},
            {3, 2, "Kepler"},
            {3, 5, "Kepler"},
            {3, 7, "Kepler"},
            {5, 0, "Maxwell"},
            {5, 2, "Maxwell"},
            {6, 0, "Pascal"},
            {6, 1, "Pascal"},
            {6, 2, "Pascal"},
            {7, 0, "Volta"},
            {7, 2, "Volta"},
            {7, 5, "Turing"},
            {8, 0, "Ampere"},
            {8, 6, "Ampere"},
            {8, 7, "Ampere"},
            {8, 9, "Ada Lovelace"},
            {9, 0, "Hopper"},
            {9, 4, "Hopper"}
        };

        for (auto &arch : architectures) {
            if (arch.major == major && arch.minor == minor) {
                return arch.name;
            }
        }

        return "Unknown Architecture";
    }

    void printWarpDetails(const cudaDeviceProp &prop) {
        std::cout << "\n--- Warp Characteristics ---" << std::endl;
        std::cout << "Warp Size: " << prop.warpSize << " threads" << std::endl;
        
        int maxActiveThreadsPerSM = prop.maxThreadsPerMultiProcessor;
        int maxActiveWarpsPerSM = maxActiveThreadsPerSM / prop.warpSize;
        
        std::cout << "Max Active Warps per SM: " << maxActiveWarpsPerSM << std::endl;
    }

    void printDeviceCapabilities(const cudaDeviceProp &prop) {
        std::cout << "\n--- Advanced Capabilities ---" << std::endl;
        
        std::cout << "Compute Capabilities:" << std::endl;
        std::cout << " - Async Engine Count: " << prop.asyncEngineCount << std::endl;
        std::cout << " - Unified Addressing: " 
                  << (prop.unifiedAddressing ? "Supported" : "Not Supported") << std::endl;
        std::cout << " - Managed Memory: " 
                  << (prop.managedMemory ? "Supported" : "Not Supported") << std::endl;
        
        std::cout << " - Concurrent Kernel Execution: " 
                  << (prop.concurrentKernels ? "Supported" : "Not Supported") << std::endl;
        std::cout << " - Cooperative Launch: " 
                  << (prop.cooperativeLaunch ? "Supported" : "Not Supported") << std::endl;
    }

    void printMemoryDetails(const cudaDeviceProp &prop) {
        std::cout << "\n--- Advanced Memory Specifications ---" << std::endl;
        
        std::cout << "Memory Types:" << std::endl;
        std::cout << " - Surface Memory: " 
                  << (prop.surfaceAlignment ? "Supported" : "Not Supported") << std::endl;
        std::cout << " - Texture Memory: " 
                  << (prop.textureAlignment ? "Supported" : "Not Supported") << std::endl;
        
        std::cout << "\nCache Specifications:" << std::endl;
        std::cout << " - L2 Cache Size: " 
                  << prop.l2CacheSize / 1024 << " KB" << std::endl;
    }

    void printNumericalCapabilities(const cudaDeviceProp &prop) {
        std::cout << "\n--- Numerical Compute Capabilities ---" << std::endl;
        std::cout << "Double Precision Performance: " 
                  << (prop.major >= 2 ? "Supported" : "Limited") << std::endl;
        std::cout << "Tensor Cores: " 
                  << (prop.major >= 7 ? "Available" : "Not Available") << std::endl;
    }

public:
    void inspectDevices() {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);

        for (int device = 0; device < deviceCount; ++device) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, device);

            std::cout << "\n===== CUDA DEVICE " << device << " COMPREHENSIVE REPORT =====" << std::endl;
            
            std::cout << "Device Name: " << prop.name << std::endl;
            std::cout << "Architecture: " 
                      << getArchitectureName(prop.major, prop.minor) 
                      << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;

            std::cout << "Total Global Memory: " 
                      << std::fixed << std::setprecision(2) 
                      << prop.totalGlobalMem / BYTES_PER_GIGABYTE << " GB" << std::endl;

            printWarpDetails(prop);
            printDeviceCapabilities(prop);
            printMemoryDetails(prop);
            printNumericalCapabilities(prop);
        }
    }
};

int main() {
    CUDADeviceInspector inspector;
    
    try {
        inspector.inspectDevices();
    } catch (const std::exception& e) {
        std::cerr << "Error during device inspection: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}