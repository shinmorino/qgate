#pragma once

#include <cuda_runtime_api.h>
#include "DeviceTypes.h"
#include "SimpleMemoryStore.h"

namespace qgate_cuda {

class CUDADevice {
public:
    enum { hTmpMemBufSize = 1 << 28 };
    enum { dTmpMemBufSize = 1 << 20 };

    CUDADevice();

    ~CUDADevice();

    int getDeviceNumber() const { return devNo_; }

    void initialize(int devNo);
    
    void finalize();

    size_t getMemSize() const {
        return devProp_.totalGlobalMem;
    }
    
    int getNumSMs() const {
        return devProp_.multiProcessorCount;
    }

    void makeCurrent();

    void checkCurrentDevice();

    void synchronize();

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

    SimpleMemoryStore &tempHostMemory() {
        return hostMemStore_;
    }

    SimpleMemoryStore &tempDeviceMemory() {
        return deviceMemStore_;
    }

    /* FIXME: add synchronize() */
    
private:
    void *h_buffer_;
    void *d_buffer_;
    SimpleMemoryStore deviceMemStore_;
    SimpleMemoryStore hostMemStore_;
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
