#pragma once

#include "DeviceTypes.h"

namespace qgate_cuda {

class DeviceSum {
public:
    DeviceSum();
    ~DeviceSum();

    void prepare();

    void finalize();
    
    template<class V, class F>
    V operator()(qgate::QstateIdxType begin, qgate::QstateIdxType end, const F &f);

    template<class V>
    V *getHostMem() {
        return static_cast<V*>(h_partialSum_);
    }
    
    void *h_partialSum_;
    int nBlocks_;
};

}
