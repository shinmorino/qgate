#pragma once

#include "Types.h"
#include "CUDADevice.h"
#include "MultiChunkPtr.h"
#include <set>
#include <map>

namespace qgate_cuda {


struct DeviceChunk {
    CUDADevice *device;
    void *ptr;
};


class DeviceCachedMemoryStore {
public:

    DeviceCachedMemoryStore();
    ~DeviceCachedMemoryStore();

    void setDevice(CUDADevice *device, qgate::QstateSize memStoreSizeOverride);
    
    void releaseAllChunks();
    
    qgate::QstateSize getNAvailableChunks(int po2idx, bool includingCachedChunks) const;

    int tryReserveChunks(int po2idx, int nChunks);

    bool tryReserveChunk(int po2idx);
    
    bool allocate(DeviceChunk *chunk, int po2idx);

    void deallocate(DeviceChunk &chunk, int po2idx);
    
private:
    enum { deviceMemoryMargin = 256 << 20 };
    
    typedef std::set<void *> ChunkSet;
    bool allocateCachedChunk(int po2idx);
    bool hasCachedChunk(int po2idx) const;
    void releaseCachedChunk(int po2idx);
    qgate::QstateSize getFreeSize() const;

    CUDADevice *device_;

    typedef std::map<int, ChunkSet> ChunkStore;
    ChunkStore allocated_;
    ChunkStore cached_;
    qgate::QstateSize memStoreSizeOverride_;
};



class MultiDeviceChunk {
public:
    MultiDeviceChunk(int po2idx) {
        po2idx_ = po2idx;
        nChunks_ = 0;
    }

    template<class V>
    MultiChunkPtr<V> getMultiChunkPtr() const;

    int getNChunks() const {
        return nChunks_;
    }

    int getPo2Idx() const {
        return po2idx_;
    }

    DeviceChunk &get(int idx) {
        assert(nChunks_ < MAX_N_CHUNKS);
        return chunks_[idx];
    }

    const DeviceChunk &get(int idx) const {
        assert(nChunks_ < MAX_N_CHUNKS);
        return chunks_[idx];
    }

    void add(DeviceChunk &chunk) {
        chunks_[nChunks_] = chunk;
        ++nChunks_;
        assert(nChunks_ <= MAX_N_CHUNKS);
    }

private:
    DeviceChunk chunks_[MAX_N_CHUNKS];
    int nChunks_;
    int po2idx_;

    /* hidden c-tor */
    MultiDeviceChunk(const MultiDeviceChunk &);
};


template<class V>
MultiChunkPtr<V> MultiDeviceChunk::getMultiChunkPtr() const {
    MultiChunkPtr<V> ptr;
    for (int idx = 0; idx < nChunks_; ++idx)
        ptr.d_ptrs[idx] = (V*)chunks_[idx].ptr;
    int nLanesInChunk = po2idx_ - ((sizeof(V) / 8) + 2);
    ptr.setNLanesInChunk(nLanesInChunk);
    return ptr;
}


class MultiDeviceMemoryStore {
public:

    MultiDeviceMemoryStore();

    ~MultiDeviceMemoryStore();

    void initialize(CUDADevices &devices,
                    int maxPo2idxPerChunk, qgate::QstateSize deviceMemorySize);

    void finalize();
    
    MultiDeviceChunk *allocate(int po2idx);

    void deallocate(MultiDeviceChunk *mchunk);


private:
    MultiDeviceChunk *allocateOneChunk(int po2idx);
    MultiDeviceChunk *allocateChunksInOneDevice(int po2idx, int nRequestedChunks);
    MultiDeviceChunk *allocateChunksBalanced(int nChunksPerDevice,
                                             int po2idx, int nRequestedChunks);
    MultiDeviceChunk *allocateChunksUnbalanced(int nChunksPerDevice,
                                               int po2idx, int nRequestedChunks);

    DeviceCachedMemoryStore *memStoreList_;
    int nStores_;
    int maxPo2idxPerChunk_;
};

}
