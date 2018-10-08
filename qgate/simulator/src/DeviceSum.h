#pragma once

#include "DeviceTypes.h"

namespace cuda_runtime {

class DeviceSum {
public:
    DeviceSum();
    ~DeviceSum();

    void prepare();

    void finalize();
    
    template<class F>
    real operator()(QstateIdxType begin, QstateIdxType end, const F &f);
    
    real *h_partialSum_;
    int nBlocks_;
};

}
