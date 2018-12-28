#pragma once

namespace qgate_cuda {

template<class V>
struct MultiChunkPtr {

    void setNLanesInChunk(int _nLanesInChunk) {
        nLanesInChunk = _nLanesInChunk;
        mask = (qgate::Qone << nLanesInChunk) - 1;
    }

    V *getPtr(qgate::QstateIdx idx) {
        int devIdx = int(idx >> nLanesInChunk);
        qgate::QstateIdx idxInDev = idx & mask;
        return &d_ptrs[devIdx][idxInDev];
    }

#ifdef __NVCC__
    __device__ __forceinline__
    V &operator[](qgate::QstateIdx idx) {
        int devIdx = int(idx >> nLanesInChunk);
        qgate::QstateIdx idxInDev = idx & mask;
        return d_ptrs[devIdx][idxInDev];
    }

    __device__ __forceinline__
    const V &operator[](qgate::QstateIdx idx) const {
        int devIdx = int(idx >> nLanesInChunk);
        qgate::QstateIdx idxInDev = idx & mask;
        return d_ptrs[devIdx][idxInDev];
    }
#endif
    
    V *d_ptrs[MAX_N_CHUNKS];
    int nLanesInChunk;
    qgate::QstateIdx mask;
};


}

