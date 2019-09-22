#include "CUDADevice.h"
#include <algorithm>
#include <list>

using namespace qgate_cuda;

namespace {

#ifdef __GNUC__
__thread int currentDevNo_ = -1;
#endif

#ifdef _MSC_VER
__declspec(thread) int currentDevNo_ = -1;
#endif
}


CUDADevice::CUDADevice() {
    h_buffer_ = NULL;
    d_buffer_ = NULL;
}

CUDADevice::~CUDADevice() {
    finalize();
}

void CUDADevice::initialize(int devIdx, int devNo) {
    devIdx_ = devIdx;
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
    makeCurrent();
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
    if (::currentDevNo_ != devNo_) {
        throwOnError(cudaSetDevice(devNo_));
        ::currentDevNo_ = devNo_;
    }
}

void CUDADevice::checkCurrentDevice() {
    throwErrorIf(::currentDevNo_ != devNo_,
                 "Device(%d) is not current(%d).", devNo_, ::currentDevNo_);
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


CUDADeviceList qgate_cuda::unique(CUDADeviceList &_devices) {
    /* remove duplicates in devices. */
    auto lessDeviceNumber = [](const CUDADevice *dev0, const CUDADevice *dev1) {
        return dev0->getDeviceIdx() < dev1->getDeviceIdx();
    };
    CUDADeviceList devices(_devices);
    std::sort(devices.begin(), devices.end(), lessDeviceNumber);
    auto eqDeviceNumber = [](const CUDADevice *dev0, const CUDADevice *dev1) {
        return dev0->getDeviceIdx() == dev1->getDeviceIdx();
    };
    auto duplicates = std::unique(devices.begin(), devices.end(), eqDeviceNumber);
    devices.erase(duplicates, devices.end());
    return devices;
}

/* CUDADevices */

CUDADevices::CUDADevices() {
}

CUDADevices::~CUDADevices() {
}

void CUDADevices::probe() {
    try {
        int count = 0;
        throwOnError(cudaGetDeviceCount(&count));
        if (count == 0)
            throwError("No CUDA device found.");
        
        /* device topology map*/
        deviceTopoMap_.resize(count);
        
        for (int devIdx = 0; devIdx < count; ++devIdx) {
            throwOnError(cudaSetDevice(devIdx));
            deviceTopoMap_[devIdx].resize(count);
            for (int peerIdx = 0; peerIdx < count; ++peerIdx) {
                if (devIdx != peerIdx) {
                    int canAccessPeer = 0;
                    throwOnError(cudaDeviceCanAccessPeer(&canAccessPeer, devIdx, peerIdx));
                    deviceTopoMap_[devIdx][peerIdx] = canAccessPeer;
                    if (canAccessPeer)
                        throwOnError(cudaDeviceEnablePeerAccess(peerIdx, 0));
                }
                else {
                    deviceTopoMap_[devIdx][peerIdx] = 1;
                }
            }
        }
    }
    catch (...) {
        deviceTopoMap_.clear();
        throw;
    }
}

void CUDADevices::create(const qgate::IdList &_deviceNos) {

    qgate::IdList deviceNos;
    if (_deviceNos.empty()) {
        deviceNos = extractDeviceCluster();
    }
    else {
        deviceNos = _deviceNos;
        /* deviceNos are given.  Cheking input. */
        for (int idx = 0; idx < (int)deviceNos.size(); ++idx) {
            int devNo = deviceNos[idx];
            if ((int)deviceTopoMap_.size() <= devNo)
                throwError("device[%d] not found.", devNo);
        }
    }
    
    assert(!deviceNos.empty());
    
    for (int idx = 0; idx < (int)deviceNos.size(); ++idx) {
        CUDADevice *device = new CUDADevice();
        int devNo = deviceNos[idx];
        device->initialize(idx, devNo);
        devices_.push_back(device);
    }
    
    /* reset current device */
    ::currentDevNo_ = -1;
}

void CUDADevices::clear() {
    for (int idx = 0; idx < (int)devices_.size(); ++idx) {
        CUDADevice *device = devices_[idx];
        delete device;
    }
    devices_.clear();
}

void CUDADevices::synchronize() {
    for (auto *device : devices_)
        device->synchronize();
}

void CUDADevices::finalize() {
    /* make sure all device instances are deleted. */
    clear();
    deviceTopoMap_.clear();

    /* not checking errors */ 
    int count = 0;
    cudaGetDeviceCount(&count);
    for (int devIdx = 0; devIdx < count; ++devIdx) {
        cudaSetDevice(devIdx);
        cudaDeviceReset();
    }
}

qgate::IdList CUDADevices::extractDeviceCluster() const {

    /* creating default list */
    qgate::IdList deviceNos;
    for (int idx = 0; idx < (int)deviceTopoMap_.size(); ++idx)
        deviceNos.push_back(idx);

    /* finding clusters according to topology. */
    qgate::IdListList clusters;
    std::list<int> devs(deviceNos.begin(), deviceNos.end());
    while (!devs.empty()) {
        int devNo = devs.front();
        devs.pop_front();
        const qgate::IdList &peers = deviceTopoMap_[devNo];
        qgate::IdList cluster;
        cluster.push_back(devNo);
        for (auto devIt = devs.begin(); devIt != devs.end(); ) {
            int peerDevNo = *devIt;
            if (peers[peerDevNo] != 0) {
                cluster.push_back(peerDevNo);
                devIt = devs.erase(devIt);
            }
            else {
                ++devIt;
            }
        }
        clusters.push_back(cluster);
    }

    /* select the best cluster */
    std::vector<size_t> memSizeList, freeSizeList;
    for (int clusterIdx = 0; clusterIdx < (int)clusters.size(); ++clusterIdx) {
        size_t totalMemSize = 0, totalFreeSize = 0;
        const qgate::IdList &cluster = clusters[clusterIdx];
        for (int devIdx = 0; devIdx < (int)cluster.size(); ++devIdx) {
            int devNo = cluster[devIdx];
            throwOnError(cudaSetDevice(devNo));
            size_t free, total;
            throwOnError(cudaMemGetInfo(&free, &total));
            totalMemSize += total;
            totalFreeSize += free;
        }
        memSizeList.push_back(totalMemSize);
        freeSizeList.push_back(totalFreeSize);
    }

    size_t biggestMemSize = 0, biggestFreeSize = 0;
    int bestClusterIdx = 0;
    for (int clusterIdx = 0; clusterIdx < (int)clusters.size(); ++clusterIdx) {
        size_t free = freeSizeList[clusterIdx], total = memSizeList[clusterIdx];
        if (biggestMemSize < total) {
            biggestMemSize = total;
            biggestFreeSize = free;
            bestClusterIdx = clusterIdx;
        }
        else if (biggestMemSize == total) {
            if (biggestFreeSize < free) {
                biggestFreeSize = free;
                bestClusterIdx = clusterIdx;
            }
        }
    }
    return clusters[bestClusterIdx];
}


qgate::QstateSize CUDADevices::getMinDeviceMemorySize() const {
    size_t minMemSize = 1LL << 62;
    for (int idx = 0; idx < (int)devices_.size(); ++idx) {
        CUDADevice *device = devices_[idx];
        minMemSize = std::min(device->getMemSize(), minMemSize);
    }
    return minMemSize;
}
