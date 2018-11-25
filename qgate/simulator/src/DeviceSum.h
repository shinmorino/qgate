#pragma once

#include "CUDADevice.h"

namespace qgate_cuda {
    
template<class V>    
class DeviceSumType {
public:
    DeviceSumType();
    ~DeviceSumType();

    void prepare();

    void allocate(CUDADevice &rsrc);

    void deallocate(CUDADevice &rsrc);
    
    template<class F>
    V operator()(qgate::QstateIdx begin, qgate::QstateIdx end, const F &f);

private:
    V *h_partialSum_;
    int nBlocks_;
};

}
