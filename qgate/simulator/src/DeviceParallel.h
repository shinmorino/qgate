#pragma once

#include "DeviceTypes.h"
#include "CUDAQubitStates.h"

namespace qgate_cuda {

using qgate::QstateIdx;


template<class F>
__global__
void transformKernel(F func, qgate::QstateIdx offset, qgate::QstateIdx size) {
    QstateIdx gid = QstateIdx(blockDim.x) * blockIdx.x + threadIdx.x;
    if (gid < size)
        func(gid + offset);
}


template<class C>
void transform(qgate::QstateIdx begin, qgate::QstateIdx end, const C &functor) {
    dim3 blockDim(128);
    QstateIdx size = end - begin;
    dim3 gridDim((unsigned int)divru(size, blockDim.x));
    transformKernel<<<gridDim, blockDim>>>(functor, begin, size);
    throwOnError(cudaGetLastError());
    DEBUG_SYNC;
}


}
