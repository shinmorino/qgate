#pragma once

#include <cuda_runtime_api.h>
#include "DeviceTypes.h"

namespace qgate_cuda {

class CUDADevice {
public:
    enum { hMemBufSize = 1 << 28 };
    enum { dMemBufSize = 1 << 20 };

    CUDADevice() {
        h_buffer_ = NULL;
        d_buffer_ = NULL;
    }
    ~CUDADevice() {
        finalize();
    }
    
    void prepare() {
        throwOnError(cudaHostAlloc(&h_buffer_, hMemBufSize, cudaHostAllocPortable));
        throwOnError(cudaMalloc(&d_buffer_, dMemBufSize));
    }
    
    void finalize() {
        if (h_buffer_ != NULL)
            throwOnError(cudaFreeHost(h_buffer_));
        if (d_buffer_ != NULL)
            throwOnError(cudaFree(d_buffer_));
        h_buffer_ = NULL;
        d_buffer_ = NULL;
    }

    template<class V>
    V *getHostMem(size_t size) {
        throwErrorIf(hostMemSize<V>() < size, "Requested size too large.");
        return static_cast<V*>(h_buffer_);
    }

    template<class V>
    size_t hostMemSize() const {
        return (size_t)hMemBufSize / sizeof(V);
    }

    template<class V>
    V *getDeviceMem(size_t size) {
        throwErrorIf(deviceMemSize<V>() < size, "Requested size too large.");
        return static_cast<V*>(d_buffer_);
    }

    template<class V>
    size_t deviceMemSize() const {
        return (size_t)dMemBufSize / sizeof(V);
    }

private:
    void *h_buffer_;
    void *d_buffer_;
};

}
