#pragma once

#include "CUDADevice.h"

namespace qgate_cuda {

template<class V>
struct DeviceSum {
    DeviceSum(CUDADevice &dev) : dev_(dev) {
        nBlocks_ = 0;
    }
    
    template<class F>
    void launch(qgate::QstateIdx begin, qgate::QstateIdx end, const F &f);

    V sync() {
        dev_.synchronize(); /* FIXME: add stream. */
        V sum = V();
        for (int idx = 0; idx < nBlocks_; ++idx)
            sum += h_partialSum_[idx];
        dev_.tempHostMemory().reset();
        return sum;
    }
    
    CUDADevice &dev_;
    V *h_partialSum_;
    int nBlocks_;
};

template<class V, class F>
V deviceSum(CUDADevice &dev,
            qgate::QstateIdx begin, qgate::QstateIdx end, const F &f) {
    
    DeviceSum<V> devSum(dev);
    devSum.launch(begin, end, f);
    return devSum.sync();
}

}
