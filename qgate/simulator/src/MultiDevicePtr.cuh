#pragma once

namespace qgate_cuda {

template<class V>
struct MultiDevicePtr {
    MultiDevicePtr(V * const _d_ptrs[MAX_N_DEVICES], int _nLanesInDevice) {
        for (int idx = 0; idx < MAX_N_DEVICES; ++idx)
            d_ptrs[idx] = _d_ptrs[idx];
        nLanesInDevice = _nLanesInDevice;
        mask = (qgate::Qtwo << nLanesInDevice) - 1;
    }

    __device__ __forceinline__
    V &operator[](qgate::QstateIdx idx) {
        int devIdx = int(idx >> nLanesInDevice);
        qgate::QstateIdx idxInDev = idx & mask;
        return d_ptrs[devIdx][idxInDev];
    }

    __device__ __forceinline__
    const V &operator[](qgate::QstateIdx idx) const {
        int devIdx = int(idx >> nLanesInDevice);
        qgate::QstateIdx idxInDev = idx & mask;
        return d_ptrs[devIdx][idxInDev];
    }

    __host__
    V *getPtr(qgate::QstateIdx idx) {
        int devIdx = int(idx >> nLanesInDevice);
        qgate::QstateIdx idxInDev = idx & mask;
        return &d_ptrs[devIdx][idxInDev];
    }
    
    V *d_ptrs[MAX_N_DEVICES];
    int nLanesInDevice;
    qgate::QstateIdx mask;
};


}

