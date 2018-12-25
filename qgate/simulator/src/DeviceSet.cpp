#include "DeviceSet.h"

using namespace qgate_cuda;


DeviceSet::DeviceSet() {
}

DeviceSet::~DeviceSet() {
}

void DeviceSet::clear() {
    devices_.clear();
}
    
void DeviceSet::add(CUDADevice *device) {
    devices_.push_back(device);
}
    
void DeviceSet::synchronize() {
    for (int idx = 0; idx < (int)devices_.size(); ++idx) {
        CUDADevice *device = devices_[idx];
        device->makeCurrent();
        throwOnError(cudaDeviceSynchronize());
    }
}
