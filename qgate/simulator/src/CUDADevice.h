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

    int getDeviceIdx() const { return devIdx_; }

    void initialize(int devIdx, int devNo);
    
    void finalize();

    size_t getMemSize() const {
        return devProp_.totalGlobalMem;
    }

    size_t getFreeSize();

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
    
    int devIdx_, devNo_;
};

class CUDADevices {
    friend class CUDADevice;
public:
    CUDADevices();
    ~CUDADevices();

    CUDADevice &operator[](int idx) {
        return *devices_[idx];
    }

    void probe();

    void create(const qgate::IdList &devNos);

    void clear();
    
    void finalize();
    
    int size() const {
        return (int)devices_.size();
    }

    qgate::QstateSize getMinDeviceMemorySize() const;
    
private:
    qgate::IdList extractDeviceCluster() const;

    qgate::IdListList deviceTopoMap_;
    typedef std::vector<CUDADevice*> DeviceList;
    DeviceList devices_;
    static int currentDevNo_;
};

typedef std::vector<CUDADevice*> CUDADeviceList;

}
