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
    memset(&devQstates_, 0, sizeof(devQstates_));

    qregIdList_ = qregIdList;
    /* qreg id to lane */
    for (int idx = 0; idx < (int)qregIdList.size(); ++idx)
        devQstates_.laneToQregId[idx] = qregIdList[idx];

	deviceList_ = deviceList;
    nLanesInChunk_ = nLanesInChunk;
	/* initialize device qubit states */
	devQstates_.nLanes = (int)qregIdList_.size();
    devQstates_.nLanesInChunk = nLanesInChunk_;
    qgate::QstateSize nStatesInChunk = Qone << nLanesInChunk_;
    for (int idx = 0; idx < (int)deviceList_.size(); ++idx) {
        CUDADevice *device = deviceList_[idx];
        devQstates_.d_qStatesPtrs[idx] = device->allocate<DeviceComplex>(nStatesInChunk);
    }
}

template<class real>
void CUDAQubitStates<real>::deallocate() {
    for (int idx = 0; idx < MAX_N_CHUNKS; ++idx) {
        if (devQstates_.d_qStatesPtrs[idx] != NULL)
			deviceList_[idx]->free(devQstates_.d_qStatesPtrs[idx]);
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
