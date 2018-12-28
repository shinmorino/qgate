#include "DeviceTypes.h"
#include "CUDAQubitStates.h"
#include "CUDADevice.h"

#include <string.h>
#include <algorithm>

using namespace qgate_cuda;
using qgate::Qone;
using qgate::Qtwo;


template<class real>
CUDAQubitStates<real>::CUDAQubitStates() {
    if (sizeof(real) == sizeof(float))
        prec_ = qgate::precFP32;
    else
        prec_ = qgate::precFP64;
}

template<class real>
CUDAQubitStates<real>::~CUDAQubitStates() {
    deallocate();
}

template<class real>
void CUDAQubitStates<real>::allocate(const qgate::IdList &qregIdList,
                                     CUDADeviceList &deviceList, int nLanesInChunk) {
    qregIdList_ = qregIdList;
    deviceList_ = deviceList;
    devPtr_.setNLanesInChunk(nLanesInChunk);
    qgate::QstateSize nStatesInChunk = Qone << devPtr_.nLanesInChunk;
    for (int idx = 0; idx < (int)deviceList_.size(); ++idx) {
        CUDADevice *device = deviceList_[idx];
        device->makeCurrent();
        devPtr_.d_ptrs[idx] = device->allocate<DeviceComplex>(nStatesInChunk);
    }
}

template<class real>
void CUDAQubitStates<real>::deallocate() {
    for (int idx = 0; idx < MAX_N_CHUNKS; ++idx) {
        if (devPtr_.d_ptrs[idx] != NULL)
            deviceList_[idx]->free(devPtr_.d_ptrs[idx]);
    }
    qregIdList_.clear();
}

template<class real>
int CUDAQubitStates<real>::getLane(int qregId) const {
    typename qgate::IdList::const_iterator it =
            std::find(qregIdList_.begin(), qregIdList_.end(), qregId);
    assert(it != qregIdList_.end());
    return (int)std::distance(qregIdList_.begin(), it);
}

template class CUDAQubitStates<float>;
template class CUDAQubitStates<double>;
