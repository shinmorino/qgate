#include "CUDADevice.h"

using namespace qgate_cuda;

int CUDADevice::currentDevNo_ = -1;

void CUDADevice::initialize(int devNo) {
    devNo_ = devNo;
    throwOnError(cudaSetDevice(devNo_));

    cudaDeviceProp prop;
    throwOnError(cudaGetDevice(&devNo));
    throwOnError(cudaGetDeviceProperties(&prop, devNo));
    nMaxActiveBlocksInDevice_ = prop.multiProcessorCount * 64;
    
    throwOnError(cudaHostAlloc(&h_buffer_, hTmpMemBufSize, cudaHostAllocPortable));
    throwOnError(cudaMalloc(&d_buffer_, dTmpMemBufSize));
}

void CUDADevice::makeCurrent() {
    if (currentDevNo_ != devNo_) {
        throwOnError(cudaSetDevice(devNo_));
        devNo_ = currentDevNo_;
    }
}

void CUDADevice::checkCurrentDevice() {
    throwErrorIf(currentDevNo_ != devNo_, "Device(%d) is not current(%d).", devNo_, currentDevNo_);
}

void CUDADevice::allocate(void **pv, size_t size) {
    checkCurrentDevice();
    throwOnError(cudaMalloc(&pv, size));
}

void CUDADevice::free(void *pv) {
    checkCurrentDevice();
    throwOnError(cudaFree(pv));
}

void CUDADevice::hostAllocate(void **pv, size_t size) {
    checkCurrentDevice();
    throwOnError(cudaMallocHost(&pv, size, cudaHostAllocPortable));
}

void CUDADevice::hostFree(void *pv) {
    checkCurrentDevice();
    throwOnError(cudaFree(pv));
}
