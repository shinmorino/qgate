#include "DeviceSum.h"

using namespace cuda_runtime;

DeviceSum::DeviceSum() {
    h_partialSum_ = NULL;
    nBlocks_ = -1;
}

DeviceSum::~DeviceSum() {
    finalize();
}

void DeviceSum::prepare() {
    int devNo;
    cudaDeviceProp prop;
    throwOnError(cudaGetDevice(&devNo));
    throwOnError(cudaGetDeviceProperties(&prop, devNo));
    nBlocks_ = prop.multiProcessorCount * (2048 / 128) * 4;
    throwOnError(cudaHostAlloc(&h_partialSum_, sizeof(real) * nBlocks_, cudaHostAllocPortable));
}

void DeviceSum::finalize() {
    if (h_partialSum_ != NULL)
        throwOnError(cudaFreeHost(h_partialSum_));
    h_partialSum_ = NULL;
}
