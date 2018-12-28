#pragma once

#include <cuda_runtime_api.h>
#include "DeviceTypes.h"
#include "SimpleMemoryStore.h"

namespace qgate_cuda {

class CUDADevice {
public:
    enum { hTmpMemBufSize = 1 << 28 };
    enum { dTmpMemBufSize = 1 << 20 };

    CUDADevice() {
        h_buffer_ = NULL;
        d_buffer_ = NULL;
    }
    ~CUDADevice() {
        finalize();
    }

    int getDeviceNumber() const { return devNo_; }

    void initialize(int devNo);
    
    void finalize();

    size_t getMemSize() const {
        return devProp_.totalGlobalMem;
    }
    
    void makeCurrent();

    void checkCurrentDevice();
    
    void allocate(void **pv, size_t size);

    void free(void *pv);
    
    template<class V>
    V *allocate(size_t size) {
        V *pv = NULL;
        allocate((void**)&pv, size * sizeof(V));
        return pv;
    }

    void hostAllocate(void **pv, size_t size);

    void hostFree(void *pv);
    
    template<class V>
    V *hostAllocate(size_t size) {
        V *pv = NULL;
        hostAllocate((void**)&pv, size * sizeof(V));
        return pv;
    }

    SimpleMemoryStore tempHostMemory() {
        return SimpleMemoryStore(h_buffer_, hTmpMemBufSize);
    }

    SimpleMemoryStore tempDeviceMemory() {
        return SimpleMemoryStore(d_buffer_, dTmpMemBufSize);
    }

    /* FIXME: add synchronize() */
    
private:
    void *h_buffer_;
    void *d_buffer_;
    int nMaxActiveBlocksInDevice_;
    cudaDeviceProp devProp_;
    
    int devNo_;
    static int currentDevNo_;
};

class CUDADevices {
public:
    CUDADevices();
    ~CUDADevices();

    CUDADevice &operator[](int idx) {
        return *devices_[idx];
    }

    void probe();

    void finalize();
    
    int size() const {
        return (int)devices_.size();
    }

    int maxNLanesInDevice() const;
    
private:
    typedef std::vector<CUDADevice*> DeviceList;
    DeviceList devices_;
};

typedef std::vector<CUDADevice*> CUDADeviceList;

}
