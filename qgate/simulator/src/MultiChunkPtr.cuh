#pragma once

namespace qgate_cuda {

template<class V>
struct MultiChunkPtr {
    MultiChunkPtr(V * const _d_ptrs[MAX_N_CHUNKS], int _nLanesInChunk) {
        for (int idx = 0; idx < MAX_N_CHUNKS; ++idx)
            d_ptrs[idx] = _d_ptrs[idx];
        nLanesInChunk = _nLanesInChunk;
        mask = (qgate::Qone << nLanesInChunk) - 1;
    }

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

    __host__
    V *getPtr(qgate::QstateIdx idx) {
        int devIdx = int(idx >> nLanesInChunk);
        qgate::QstateIdx idxInDev = idx & mask;
        return &d_ptrs[devIdx][idxInDev];
    }
    
    V *d_ptrs[MAX_N_CHUNKS];
    int nLanesInChunk;
    qgate::QstateIdx mask;
};


}

