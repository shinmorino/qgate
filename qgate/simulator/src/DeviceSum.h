#pragma once

#include "DeviceTypes.h"
#include "CUDAResource.h"

namespace qgate_cuda {
    
template<class V>    
class DeviceSumType {
public:
    DeviceSumType();
    ~DeviceSumType();

    void prepare();

    void allocate(CUDAResource &rsrc);

    void deallocate(CUDAResource &rsrc);
    
    template<class F>
    V operator()(qgate::QstateIdxType begin, qgate::QstateIdxType end, const F &f);

private:
    V *h_partialSum_;
    int nBlocks_;
};

}
