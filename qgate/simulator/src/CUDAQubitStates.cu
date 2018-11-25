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
void DeviceQubitStates<real>::allocate(const qgate::IdList &qregIdList, CUDADevice &device) {
    deallocate(device);

    nQregIds_ = (int)qregIdList.size();
    nStates_ = Qone << nQregIds_;
    size_t qregIdListSize = sizeof(int) * nQregIds_;
    d_qregIdList_ = device.allocate<int>(qregIdListSize);
    throwOnError(cudaMemcpy(d_qregIdList_, qregIdList.data(), qregIdListSize, cudaMemcpyDefault));
    d_qstates_ = device.allocate<DeviceComplex>(nStates_);
}

template<class real>
void DeviceQubitStates<real>::deallocate(CUDADevice &device) {
    if (d_qregIdList_ != NULL)
        device.free(d_qregIdList_);
    if (d_qstates_ != NULL)
        device.free(d_qstates_);
    d_qregIdList_ = NULL;
    d_qstates_ = NULL;
}

template<class real>
void DeviceQubitStates<real>::reset() {
    throwOnError(cudaMemset(d_qstates_, 0, sizeof(DeviceComplex) * nStates_));
    DeviceComplex cQone(1.);
    throwOnError(cudaMemcpyAsync(d_qstates_, &cQone, sizeof(DeviceComplex), cudaMemcpyDefault));
}

template<class real>
qgate::QstateIdx DeviceQubitStates<real>::getNStates() const {
    return Qone << nQregIds_;
}


template<class real>
CUDAQubitStates<real>::CUDAQubitStates(CUDADevice *device) {
    if (sizeof(real) == sizeof(float))
        prec_ = qgate::precFP32;
    else
        prec_ = qgate::precFP64;
    /* set device */
    device_ = device;
}

template<class real>
CUDAQubitStates<real>::~CUDAQubitStates() {
    deallocate();
}

template<class real>
void CUDAQubitStates<real>::allocate(const qgate::IdList &qregIdList) {
    qregIdList_ = qregIdList;
    devQstates_.allocate(qregIdList, *device_);
    devQstates_.reset();
}
    
template<class real>
void CUDAQubitStates<real>::deallocate() {
    devQstates_.deallocate(*device_);
}

template<class real>
void CUDAQubitStates<real>::reset() {
    devQstates_.reset();
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
