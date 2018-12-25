#pragma once

#include "CUDADevice.h"
#include <vector>

namespace qgate_cuda {

class DeviceSet {
public:
    DeviceSet();

    ~DeviceSet();

    void clear();
    
    void add(CUDADevice *device);
    
    CUDADevice &operator[](int idx) {
        return *devices_[idx];
    }

    int size() const {
        return (int)devices_.size();
    }
    
    void synchronize();
    
private:
    typedef std::vector<CUDADevice*> Devices;
    Devices devices_;
};

}
