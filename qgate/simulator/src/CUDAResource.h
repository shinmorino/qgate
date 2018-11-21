#include "DeviceSum.h"

namespace qgate_cuda {

struct CUDAResource {
    enum { hMemBufSize = 1 << 28 };
    enum { dMemBufSize = 1 << 20 };

    CUDAResource() {
        h_buffer_ = NULL;
        d_buffer_ = NULL;
    }
    ~CUDAResource() {
        finalize();
    }
    
    void prepare() {
        deviceSum_.prepare();
        throwOnError(cudaHostAlloc(&h_buffer_, hMemBufSize, cudaHostAllocPortable));
        throwOnError(cudaMalloc(&d_buffer_, dMemBufSize));
    }
    
    void finalize() {
        deviceSum_.finalize();
        if (h_buffer_ != NULL)
            throwOnError(cudaFreeHost(h_buffer_));
        if (d_buffer_ != NULL)
            throwOnError(cudaFree(d_buffer_));
        h_buffer_ = NULL;
        d_buffer_ = NULL;
    }

    template<class V>
    V *getHostMem() {
        return static_cast<V*>(h_buffer_);
    }

    template<class V>
    size_t hostMemSize() const {
        return (size_t)hMemBufSize / sizeof(V);
    }

    template<class V>
    V *getDeviceMem() {
        return static_cast<V*>(d_buffer_);
    }

    DeviceSum deviceSum_;
    void *h_buffer_;
    void *d_buffer_;
};

}
