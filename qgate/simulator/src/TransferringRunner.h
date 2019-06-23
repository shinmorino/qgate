#pragma once

#include "DeviceTypes.h"
#include "CUDADevice.h"
#include <vector>

namespace qgate_cuda {

/* instantiated per device */
struct  DeviceWorker {

    DeviceWorker(CUDADevice *device) : device_(device) { }
    
    virtual ~DeviceWorker() { }
    
    virtual void run(void *dst, qgate::QstateIdx begin, qgate::QstateIdx end) = 0;

    CUDADevice &getDevice() { return *device_; }

private:    
    CUDADevice *device_;
};

typedef std::vector<DeviceWorker*> DeviceWorkers;

template<class V>
void run_d2h(V *array, DeviceWorkers &devWorkers,
             qgate::QstateIdx begin, qgate::QstateIdx end);

}
