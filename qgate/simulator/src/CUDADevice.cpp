#include "CUDADevice.h"
#include <algorithm>

using namespace qgate_cuda;

int CUDADevice::currentDevNo_ = -1;


CUDADevice::CUDADevice() {
    h_buffer_ = NULL;
    d_buffer_ = NULL;
}

CUDADevice::~CUDADevice() {
    finalize();
}

void CUDADevice::initialize(int devNo) {
    devNo_ = devNo;
    makeCurrent(); /* call cudaSetDevice() and mark this device current. */

    throwOnError(cudaGetDevice(&devNo));
    throwOnError(cudaGetDeviceProperties(&devProp_, devNo));
    nMaxActiveBlocksInDevice_ = devProp_.multiProcessorCount * 64;
    
    throwOnError(cudaHostAlloc(&h_buffer_, hTmpMemBufSize, cudaHostAllocPortable));
    throwOnError(cudaMalloc(&d_buffer_, dTmpMemBufSize));
    hostMemStore_.set(h_buffer_, hTmpMemBufSize);
    deviceMemStore_.set(d_buffer_, dTmpMemBufSize);
}

void CUDADevice::finalize() {
    if (h_buffer_ != NULL)
        throwOnError(cudaFreeHost(h_buffer_));
    if (d_buffer_ != NULL)
        throwOnError(cudaFree(d_buffer_));
    h_buffer_ = NULL;
    d_buffer_ = NULL;
}

size_t CUDADevice::getFreeSize() {
    makeCurrent();
    size_t free, total;
    throwOnError(cudaMemGetInfo(&free, &total));
    return free;
}

void CUDADevice::makeCurrent() {
    if (currentDevNo_ != devNo_) {
        throwOnError(cudaSetDevice(devNo_));
        currentDevNo_ = devNo_;
    }
}

void CUDADevice::checkCurrentDevice() {
    throwErrorIf(currentDevNo_ != devNo_, "Device(%d) is not current(%d).", devNo_, currentDevNo_);
}

void CUDADevice::synchronize() {
    makeCurrent();
    throwOnError(cudaDeviceSynchronize());
}


void CUDADevice::allocate(void **pv, size_t size) {
    makeCurrent();
    throwOnError(cudaMalloc(pv, size));
}

void CUDADevice::free(void *pv) {
    makeCurrent();
    throwOnError(cudaFree(pv));
}

void CUDADevice::hostAllocate(void **pv, size_t size) {
    makeCurrent();
    throwOnError(cudaMallocHost(pv, size, cudaHostAllocPortable));
}

void CUDADevice::hostFree(void *pv) {
    makeCurrent();
    throwOnError(cudaFree(pv));
}


/* CUDADevices */
CUDADevices::CUDADevices() {
}

CUDADevices::~CUDADevices() {
}



void CUDADevices::probe() {
    int count = 0;
    throwOnError(cudaGetDeviceCount(&count));

    try {
        for (int idx = 0; idx < count; ++idx) {
            CUDADevice *device = new CUDADevice();
            devices_.push_back(device);
            device->initialize(idx);
        }
    }
    catch (...) {
        finalize();
        throw;
    }
}

void CUDADevices::finalize() {
    for (int idx = 0; idx < (int)devices_.size(); ++idx)
        delete devices_[idx];
    devices_.clear();
}


int CUDADevices::maxNLanesInDevice() const {
    size_t memSize = 1LL << 62;
    for (int idx = 0; idx < (int)devices_.size(); ++idx) {
        CUDADevice *device = devices_[idx];
        memSize = std::min(device->getMemSize(), memSize);
    }
    int nLanesInDevice = 63;
    size_t minSizePo2;
    do {
        --nLanesInDevice;
        minSizePo2 = 1LL << nLanesInDevice;
    } while (memSize < minSizePo2);
    return nLanesInDevice;
}
