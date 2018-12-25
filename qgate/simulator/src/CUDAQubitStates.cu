#include "DeviceTypes.h"
#include "DeviceParallel.h"
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
}

template<class real>
void CUDAQubitStates<real>::allocate(const qgate::IdList &qregIdList,
                                     DeviceSet &deviceSet, int nLanesInDevice) {
    memset(&devQstates_, 0, sizeof(devQstates_));

    qregIdList_ = qregIdList;
    /* qreg id to lane */
    for (int idx = 0; idx < (int)qregIdList.size(); ++idx)
        devQstates_.qregIdToLane[qregIdList[idx]] = idx;

    nLanesInDevice_ = nLanesInDevice;
    qgate::QstateSize nStatesInDevice = Qone << nLanesInDevice;
    for (int idx = 0; idx < deviceSet.size(); ++idx)
        devQstates_.d_qStatesPtrs[idx] = deviceSet[idx].allocate<DeviceComplex>(nStatesInDevice);
}

template<class real>
void CUDAQubitStates<real>::deallocate(DeviceSet &deviceSet) {
    for (int idx = 0; idx < MAX_N_DEVICES; ++idx) {
        if (devQstates_.d_qStatesPtrs[idx] != NULL)
            deviceSet[idx].free(devQstates_.d_qStatesPtrs[idx]);
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
