#pragma once

#include "Types.h"
#include "CUDAQubitStates.h"

namespace qgate {

class ProcessorRelocator {
public:

    void setLanes(const IdList &hiLanes, int loLane, int varLane);

    void setLanes(int loLane, int varLane);

    IdList generateIdxList(int nBits);
    
private:
    
    IdList masks_;
    int setBitMask_;
    int varBit_;
};

template<class real>
IdList relocateProcessors(const qgate_cuda::CUDAQubitStates<real> &cuQStates, const IdList &hiLanes, int loLane, int varLane);

template<class real>
IdList relocateProcessors(const qgate_cuda::CUDAQubitStates<real> &cuQStates, int loLane, int varLane);


}
