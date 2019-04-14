#include "MultiDeviceMemoryStore.h"
#include <algorithm>
#include <numeric>

using namespace qgate_cuda;
using qgate::Qone;
using qgate::QstateIdx;
using qgate::QstateSize;

DeviceCachedMemoryStore::DeviceCachedMemoryStore() {
}

DeviceCachedMemoryStore::~DeviceCachedMemoryStore() {
}

void DeviceCachedMemoryStore::setDevice(CUDADevice *device, QstateSize memStoreSizeOverride) {
    device_ = device;
    memStoreSizeOverride_ = memStoreSizeOverride;
}

void DeviceCachedMemoryStore::releaseAllChunks() {
    device_->makeCurrent();
    
    for (ChunkStore::iterator it = cached_.begin(); it != cached_.end(); ++it) {
        ChunkSet &chunkset = it->second;
        for (ChunkSet::iterator chunk = chunkset.begin(); chunk != chunkset.end(); ++chunk) {
            device_->free(*chunk);
        }
    }
    cached_.clear();
    
    bool hasLeak = false;
    for (ChunkStore::iterator it = allocated_.begin(); it != allocated_.end(); ++it) {
        ChunkSet &chunkset = it->second;
        for (ChunkSet::iterator chunk = chunkset.begin(); chunk != chunkset.end(); ++chunk) {
            device_->free(*chunk);
            hasLeak = true;
        }
    }
    allocated_.clear();
    if (hasLeak)
        qgate::log("Device memory leak found.");
}

QstateSize DeviceCachedMemoryStore::getFreeSize() const {
    if (memStoreSizeOverride_ == -1)
        return (QstateSize)device_->getFreeSize();
    QstateSize allocatedSize = 0;
    for (ChunkStore::const_iterator it = allocated_.begin(); it != allocated_.end(); ++it) {
        QstateSize chunkSize = Qone << it->first;
        allocatedSize += chunkSize * it->second.size();
    }
    for (ChunkStore::const_iterator it = cached_.begin(); it != cached_.end(); ++it) {
        QstateSize chunkSize = Qone << it->first;
        allocatedSize += chunkSize * it->second.size();
    }
    return memStoreSizeOverride_ - allocatedSize;
}

bool DeviceCachedMemoryStore::allocateCachedChunk(int po2idx) {
    QstateSize size = Qone << po2idx;
    if (size <= getFreeSize()) {
        device_->makeCurrent();
        try {
            void *pv;
            device_->allocate(&pv, size);
            auto res = cached_[po2idx].insert(pv);
            abortIf(!res.second, "duplicate chunk");
            return true;
        }
        catch (...) {
            return false;
        }
    }
    return false;
}

bool DeviceCachedMemoryStore::hasCachedChunk(int po2idx) const {
    ChunkStore::const_iterator it = cached_.find(po2idx);
    if (it == cached_.end())
        return false;
    return !it->second.empty();
}

void DeviceCachedMemoryStore::releaseCachedChunk(int po2idx) {
    ChunkStore::iterator it = cached_.find(po2idx);
    abortIf(it == cached_.end(), "no cached chunk.");
    ChunkSet::iterator cit = cached_[po2idx].begin();
    abortIf(cit == cached_[po2idx].end(), "no cached chunk");
    device_->makeCurrent();
    device_->free(*cit);
    cached_[po2idx].erase(cit);
}

bool DeviceCachedMemoryStore::tryReserveChunk(int po2idx) {
    device_->makeCurrent();
    if (allocateCachedChunk(po2idx))
        return true;
    
    /* try to release a chank larger than the requested size. */
    ChunkStore::const_iterator ub = cached_.upper_bound(po2idx);
    if (ub != cached_.end()) {
        releaseCachedChunk(ub->first);
        if (allocateCachedChunk(po2idx))
            return true;
        qgate::log("failed allocation though 2x~ capacity released.");
    }

    /* release smaller chunks to get free mem */
    QstateSize freeSize = getFreeSize();
    QstateSize requestedSize = Qone << po2idx;
    for (ChunkStore::const_iterator it = cached_.begin(); it != cached_.end(); ++it) {
        if (hasCachedChunk(it->first))
            releaseCachedChunk(it->first);
        QstateSize chunkSize = Qone << it->first;
        freeSize += chunkSize;
        if (requestedSize <= freeSize) {
            if (allocateCachedChunk(po2idx))
                return true;
        }
    }
    return false;
}

QstateSize DeviceCachedMemoryStore::getNAvailableChunks(int po2idx) const {
    QstateSize size = Qone << po2idx;
    QstateSize nAvailableChunks = getFreeSize() / size;
    ChunkStore::const_iterator it = cached_.find(po2idx);
    if (it != cached_.end())
        nAvailableChunks += it->second.size();
    return nAvailableChunks;
}

bool DeviceCachedMemoryStore::allocate(DeviceChunk *chunk, int po2idx) {
    ChunkSet &cachedSet = cached_[po2idx];
    if (cachedSet.empty()) {
        if (!allocateCachedChunk(po2idx))
            return false;
    }
    
    ChunkSet::iterator it = cachedSet.begin();
    void *pv = *it;
    cachedSet.erase(it);
    ChunkSet &allocatedSet = allocated_[po2idx];
    allocatedSet.insert(pv);

    chunk->ptr = pv;
    chunk->device = device_;
    return true;
}

void DeviceCachedMemoryStore::deallocate(DeviceChunk &chunk, int po2idx) {
    auto nErased = allocated_[po2idx].erase(chunk.ptr);
    abortIf(nErased == 0, "trying to free unknown ptr.");
    auto res = cached_[po2idx].insert(chunk.ptr);
    abortIf(!res.second, "duplicate ptr.");

    chunk.device = NULL;
    chunk.ptr = NULL;
}


MultiDeviceMemoryStore::MultiDeviceMemoryStore() {
    memStoreList_ = NULL;
    nStores_ = 0;
}

MultiDeviceMemoryStore::~MultiDeviceMemoryStore() {
}

void MultiDeviceMemoryStore::
initialize(CUDADevices &devices, int maxPo2idxPerChunk, QstateSize memStoreSize) {
    if (memStoreList_ != NULL)
        terminate();
    memStoreList_ = new DeviceCachedMemoryStore[devices.size()];
    nStores_ = (int)devices.size();
    for (int idx = 0; idx < nStores_; ++idx)
        memStoreList_[idx].setDevice(&devices[idx], memStoreSize);
    
    maxPo2idxPerChunk_ = maxPo2idxPerChunk;
}

void MultiDeviceMemoryStore::finalize() {
    for (int idx = 0; idx < nStores_; ++idx)
        memStoreList_[idx].releaseAllChunks();
    delete [] memStoreList_;
    memStoreList_ = NULL;
}
    
MultiDeviceChunk *MultiDeviceMemoryStore::allocate(int po2idx) {
    MultiDeviceChunk *mchunk = _allocate(po2idx);
    if (mchunk != NULL)
        return mchunk;
    if (tryReserveSpace(po2idx))
        mchunk = _allocate(po2idx);
    if (mchunk == NULL)
        throwError("Out of device memory.");
    return mchunk;
}

MultiDeviceChunk *MultiDeviceMemoryStore::_allocate(int po2idx) {
    int nRequestedChunks;
    if (maxPo2idxPerChunk_ <= po2idx) {
        nRequestedChunks = 1 << (po2idx - maxPo2idxPerChunk_);
        po2idx = maxPo2idxPerChunk_;
    }
    else {
        nRequestedChunks = 1;
    }

    /* check if we have free memory. */
    std::vector<QstateSize> nAvailableChunksList(nStores_);
    for (int devIdx = 0; devIdx < nStores_; ++devIdx)
        nAvailableChunksList[devIdx] = memStoreList_[devIdx].getNAvailableChunks(po2idx);
    QstateSize nAvailableChunks = std::accumulate(nAvailableChunksList.begin(), nAvailableChunksList.end(), QstateSize());
    if (nAvailableChunks < nRequestedChunks)
        return NULL; /* failed. */

    MultiDeviceChunk *mchunk = new MultiDeviceChunk(po2idx, nRequestedChunks);

    /* 1. try to allocate from one GPU */
    int devIdx = 0;
    for (; devIdx < nStores_; ++devIdx) {
        QstateSize nAvailableChunks = nAvailableChunksList[devIdx];
        if (nRequestedChunks <= nAvailableChunks)
            break;
    }
    if (devIdx != nStores_) {
        /* all chunks are allocatable from one Device */
        bool success = false;
        for (int idx = 0; idx < nRequestedChunks; ++idx) {
            DeviceChunk chunk;
            success = memStoreList_[devIdx].allocate(&chunk, po2idx);
            mchunk->add(chunk);
            if (!success) {
                qgate::log("Unexpectedly failed device memory allocation.");
                break;
            }
        }
        if (success)
            return mchunk;
        deallocate(mchunk);
    }
    
    /* chunk is segmented among devices. */
    while (nRequestedChunks != 0) {
        QstateSize maxNAvailableChunks = 0;
        int maxDevIdx = -1;
        for (int devIdx = 0; devIdx < nStores_; ++devIdx) {
            QstateSize nAvailableChunks = memStoreList_[devIdx].getNAvailableChunks(po2idx);
            if (maxNAvailableChunks < nAvailableChunks) {
                maxNAvailableChunks = nAvailableChunks;
                maxDevIdx = devIdx;
            }
        }
        if (maxDevIdx == -1) {
            deallocate(mchunk);
            delete mchunk;
            return NULL; /* no enough capcity */
        }
        int nToBeAllocated = (int)std::min((QstateSize)nRequestedChunks, maxNAvailableChunks);
        int nAllocated = 0;
        for (; nAllocated < nToBeAllocated; ++nAllocated) {
            DeviceChunk chunk;
            bool success = memStoreList_[maxDevIdx].allocate(&chunk, po2idx);
            mchunk->add(chunk);
            if (!success)
                break;
        }
        nRequestedChunks -= nAllocated;
    }
    return mchunk;
}

bool MultiDeviceMemoryStore::tryReserveSpace(int po2idx) {
    QstateSize nRequestedChunks;
    if (maxPo2idxPerChunk_ <= po2idx) {
        nRequestedChunks = Qone << (po2idx - maxPo2idxPerChunk_);
        po2idx = maxPo2idxPerChunk_;
    }
    else {
        nRequestedChunks = 1;
    }

    /* check if we have free memory. */
    QstateSize nAvailableChunks = 0;
    for (int devIdx = 0; devIdx < nStores_; ++devIdx)
        nAvailableChunks += memStoreList_[devIdx].getNAvailableChunks(po2idx);

    nRequestedChunks -= nAvailableChunks;
    int nReserved = 0;
    for (int devIdx = 0; devIdx < nStores_; ++devIdx) {
        if (memStoreList_[devIdx].tryReserveChunk(po2idx))
            ++nReserved;
        if (nRequestedChunks == nReserved)
            return true;
    }
    return false;
}


void MultiDeviceMemoryStore::deallocate(MultiDeviceChunk *mchunk) {
    for (int idx = 0; idx < mchunk->getNChunks(); ++idx) {
        DeviceChunk &chunk = mchunk->get(idx);
        int devIdx = chunk.device->getDeviceIdx();
        memStoreList_[devIdx].deallocate(chunk, mchunk->getPo2Idx());
    }
}