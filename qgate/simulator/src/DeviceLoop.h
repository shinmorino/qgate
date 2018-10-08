#pragma once

#include "Types.h"
#include <functional>
#include "cudafuncs.h"

namespace cuda_runtime {

template<class F>
__global__
void transformKernel(F func, QstateIdxType offset, QstateIdxType size) {
    QstateIdxType gid = QstateIdxType(blockDim.x) * blockIdx.x + threadIdx.x;
    if (gid < size)
        func(gid + offset);
}


template<class C>
void transform(QstateIdxType begin, QstateIdxType end, const C &functor) {
    dim3 blockDim(128);
    QstateIdxType size = end - begin;
    dim3 gridDim(divru(size, blockDim.x));
    transformKernel<<<gridDim, blockDim>>>(functor, begin, size);
    DEBUG_SYNC;
}

template<class C>
real deviceSum(QstateIdxType begin, QstateIdxType end, const C &functor);

}
