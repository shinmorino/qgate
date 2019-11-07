#include "MultiDeviceMemoryStore.h"
#include "CUDAGlobals.h"
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
        return (QstateSize)(device_->getFreeSize() - deviceMemoryMargin);
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

void DeviceCachedMemoryStore::releaseCachedChunk(int po2idx) {
    ChunkStore::iterator it = cached_.find(po2idx);
    abortIf(it == cached_.end(), "no cached chunk.");
    auto &cacheSet = it->second;
    abortIf(cacheSet.empty(), "no cached chunk");
    device_->makeCurrent();
    ChunkSet::iterator cit = cacheSet.begin();
    device_->free(*cit);
    cacheSet.erase(cit);
    if (cacheSet.empty())
        cached_.erase(it);
}

bool DeviceCachedMemoryStore::tryReserveChunk(int po2idx) {
    device_->makeCurrent();
    if (allocateCachedChunk(po2idx))
        return true;
    
    /* try to release a chank larger than the requested size. */
    ChunkStore::iterator ub = cached_.upper_bound(po2idx);
    if (ub != cached_.end()) {
        releaseCachedChunk(ub->first);
        if (allocateCachedChunk(po2idx))
            return true;
        abort_("failed allocation though 2x~ capacity released.");
    }

    /* release smaller chunks to get free mem */
    QstateSize freeSize = getFreeSize();
    QstateSize requestedSize = Qone << po2idx;
    while (!cached_.empty()) {
        /* get the largest chunk. */
        ChunkStore::reverse_iterator it = cached_.rbegin();
        QstateSize chunkSize = Qone << it->first;
        releaseCachedChunk(it->first);
        freeSize += chunkSize;
        if (requestedSize <= freeSize) {
            /* freeSize is an estimation.
             * make sure the device has enough free memory. */
            if (requestedSize <= getFreeSize()) {
                bool res = allocateCachedChunk(po2idx);
                abortIf(!res, "Failed to allocate chunk.");
                return true;
            }
        }
    }

    return false;
}

int DeviceCachedMemoryStore::tryReserveChunks(int po2idx, int nChunks) {
    ChunkStore::const_iterator it = cached_.find(po2idx);
    QstateSize nCurrentChunks = (it == cached_.end()) ? 0 : it->second.size();
    nCurrentChunks = std::min((QstateSize)nChunks, nCurrentChunks);
    for (; nCurrentChunks < nChunks; ++nCurrentChunks) {
        if (!tryReserveChunk(po2idx))
            break;
    }
    return (int)nCurrentChunks;
}

QstateSize
DeviceCachedMemoryStore::getNAvailableChunks(int po2idx, bool includeCachedChunks) const {
    QstateSize size = Qone << po2idx;
    QstateSize freeSize = getFreeSize();
    if (includeCachedChunks) {
        for (auto it = cached_.begin(); it != cached_.end(); ++it) {
            QstateSize chunkSize = Qone << it->first;
            freeSize += chunkSize * it->second.size();
        }
    }
    else {
        ChunkStore::const_iterator it = cached_.find(po2idx);
        if (it != cached_.end())
            freeSize += size * it->second.size();
    }
    return freeSize / size;
}

bool DeviceCachedMemoryStore::allocate(DeviceChunk *chunk, int po2idx) {
    /* FIXME: could be replaced by barrier in python layer. */
    cudaDevices.synchronize();
    
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
    /* FIXME: could be replaced by barrier in python layer. */
    cudaDevices.synchronize();
    
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
        finalize();
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
    int nRequestedChunks = 1;
    if (maxPo2idxPerChunk_ < po2idx) {
        nRequestedChunks = 1 << (po2idx - maxPo2idxPerChunk_);
        po2idx = maxPo2idxPerChunk_;
    }

    /* 1. allocating one chunk */
    if (nRequestedChunks == 1)
        return allocateOneChunk(po2idx);
    
    /* 2. try to allocate from one GPU */
    MultiDeviceChunk *mchunk = allocateChunksInOneDevice(po2idx, nRequestedChunks);
    if (mchunk != NULL)
        return mchunk;

    if (nStores_ == 1)
        return NULL; /* one device allocation failed.  */

    for (int nChunksPerDevice = qgate::divru(nRequestedChunks, nStores_);
         nChunksPerDevice < nRequestedChunks; ++nChunksPerDevice) {
    
        /* 3. try to allocate from multiple devices,
           # chunks per device are allocated from each device. */
        mchunk = allocateChunksBalanced(nChunksPerDevice, po2idx, nRequestedChunks);
        if (mchunk != NULL)
            return mchunk;
        
        /* 4. try to allocate from multiple devices,
           # chunks per device are not balanced. */
        mchunk = allocateChunksUnbalanced(nChunksPerDevice, po2idx, nRequestedChunks);
        if (mchunk != NULL)
            return mchunk;
    }
    
    throwError("Out of device memory.");
    return NULL;
}

MultiDeviceChunk *MultiDeviceMemoryStore::allocateOneChunk(int po2idx) {
    int devIdx = 0;
    for (; devIdx < nStores_; ++devIdx) {
        QstateSize nAvailableChunks =
                memStoreList_[devIdx].getNAvailableChunks(po2idx, false);
        if (1 <= nAvailableChunks)
            break;
    }
    if (devIdx == nStores_) {
        devIdx = 0;
        for (; devIdx < nStores_; ++devIdx) {
            QstateSize nAvailableChunks =
                    memStoreList_[devIdx].tryReserveChunks(po2idx, 1);
            if (1 == nAvailableChunks)
                break;
        }
    }
    if (devIdx == nStores_)
        return NULL;
    
    MultiDeviceChunk *mchunk = new MultiDeviceChunk(po2idx);
    DeviceChunk chunk;
    bool success = memStoreList_[devIdx].allocate(&chunk, po2idx);
    abortIf(!success, "unexpected failure of device memory allocation");
    mchunk->add(chunk);
    return mchunk;
}

MultiDeviceChunk *
MultiDeviceMemoryStore::allocateChunksInOneDevice(int po2idx, int nRequestedChunks) {
    int devIdx = 0;
    for (; devIdx < nStores_; ++devIdx) {
        QstateSize nAvailableChunks = memStoreList_[devIdx].getNAvailableChunks(po2idx, false);
        if (nRequestedChunks <= nAvailableChunks)
            break;
    }
    if (devIdx == nStores_) {
        devIdx = 0;
        for (; devIdx < nStores_; ++devIdx) {
            QstateSize nAvailableChunks = memStoreList_[devIdx].getNAvailableChunks(po2idx, true);
            if (nRequestedChunks <= nAvailableChunks)
                break;
        }
    }
    if (devIdx == nStores_)
        return NULL;

    int nAvailableChunks = memStoreList_[devIdx].tryReserveChunks(po2idx, nRequestedChunks);
    if (nAvailableChunks < nRequestedChunks)
        return NULL;
    
    MultiDeviceChunk *mchunk = new MultiDeviceChunk(po2idx);
    for (int idx = 0; idx < nRequestedChunks; ++idx) {
        DeviceChunk chunk;
        bool success = memStoreList_[devIdx].allocate(&chunk, po2idx);
        abortIf(!success, "unexpected failure of device memory allocation");
        mchunk->add(chunk);
    }
    return mchunk;
}

MultiDeviceChunk *MultiDeviceMemoryStore::
allocateChunksBalanced(int nChunksPerDevice, int po2idx, int nRequestedChunks) {

    qgate::IdList devIds;
    
    int nAvailableChunks = 0;
    for (int devIdx = 0; devIdx < nStores_; ++devIdx) {
        QstateSize nAvailableInDevice =
                memStoreList_[devIdx].getNAvailableChunks(po2idx, true);
        if (nChunksPerDevice <= nAvailableInDevice) {
            nAvailableChunks += nChunksPerDevice;
            devIds.push_back(devIdx);
        }
        if (nRequestedChunks <= nAvailableChunks)
            break;
    }
    if (nAvailableChunks < nRequestedChunks)
        return NULL;

    for (int idx = 0; idx < (int)devIds.size(); ++idx) {
        int devIdx = devIds[idx];
        int nReserved = memStoreList_[devIdx].tryReserveChunks(po2idx, nChunksPerDevice);
        if (nReserved != nChunksPerDevice)
            return NULL; /* reserving chunks failed. */
    }
    
    /* allocate the same number of chunks in each device. */
    MultiDeviceChunk *mchunk = new MultiDeviceChunk(po2idx);
    for (int idx = 0; idx < (int)devIds.size(); ++idx) {
        int devIdx = devIds[idx];
        DeviceChunk chunk;
        for (int idx = 0; idx < nChunksPerDevice; ++idx) {
            bool success = memStoreList_[devIdx].allocate(&chunk, po2idx);
            abortIf(!success, "unexpected failure of device memory allocation");
            mchunk->add(chunk);
        }
    }
    return mchunk;
}


MultiDeviceChunk *MultiDeviceMemoryStore::
allocateChunksUnbalanced(int nChunksPerDevice, int po2idx, int nRequestedChunks) {
    qgate::IdList devIds;
    std::vector<QstateSize> nAvailableChunksList;

    QstateSize nAvailableChunks = 0;
    for (int devIdx = 0; devIdx < nStores_; ++devIdx) {
        QstateSize nAvailablesInDevice = memStoreList_[devIdx].getNAvailableChunks(po2idx, true);
        if (nAvailablesInDevice <= nChunksPerDevice) {
            nAvailableChunks += nAvailablesInDevice;
            devIds.push_back(devIdx);
            nAvailableChunksList.push_back(nAvailableChunks);
        }
        if (nRequestedChunks <= nAvailableChunks)
            break;
    }
    if (nAvailableChunks < nRequestedChunks) {
        /* try freeing caches */
        nAvailableChunks = 0;
        devIds.clear();
        nAvailableChunksList.clear();
        
        for (int devIdx = 0; devIdx < nStores_; ++devIdx) {
            QstateSize nAvailablesInDevice =
                    memStoreList_[devIdx].getNAvailableChunks(po2idx, true);
            if ((nAvailablesInDevice != 0) && (nAvailablesInDevice <= nChunksPerDevice)) {
                devIds.push_back(devIdx);
                nAvailableChunksList.push_back(nAvailablesInDevice);
                nAvailableChunks += nAvailablesInDevice;
            }
            if (nRequestedChunks <= nAvailableChunks)
                break;
        }
    }
    if (nAvailableChunks < nRequestedChunks)
        return NULL;

    for (int idx = 0; idx < (int)devIds.size(); ++idx) {
        int devIdx = devIds[idx];
        QstateSize nToReserve = nAvailableChunksList[devIdx];
        QstateSize nReserved = memStoreList_[devIdx].tryReserveChunks(po2idx, (int)nToReserve);
        if (nToReserve != nReserved)
            return NULL; /* failed reserving chunk */
    }
    
    /* allocate chunks. */
    MultiDeviceChunk *mchunk = new MultiDeviceChunk(po2idx);
    for (int idx = 0; idx < (int)devIds.size(); ++idx) {
        int devIdx = devIds[idx];
        int nToAllocate = (int)std::min((QstateSize)nRequestedChunks, nAvailableChunksList[idx]);
        nRequestedChunks -= nToAllocate;
        for (int idx = 0; idx < nToAllocate; ++idx) {
            DeviceChunk chunk;
            bool success = memStoreList_[devIdx].allocate(&chunk, po2idx);
            abortIf(!success, "unexpected failure of device memory allocation");
            mchunk->add(chunk);
        }
    }
    return mchunk;
}

void MultiDeviceMemoryStore::deallocate(MultiDeviceChunk *mchunk) {
    for (int idx = 0; idx < mchunk->getNChunks(); ++idx) {
        DeviceChunk &chunk = mchunk->get(idx);
        int devIdx = chunk.device->getDeviceIdx();
        memStoreList_[devIdx].deallocate(chunk, mchunk->getPo2Idx());
    }
}
