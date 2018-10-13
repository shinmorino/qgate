#include "DeviceTypes.h"
#include "DeviceLoop.h"
#include "CUDARuntime.h"

#include <string.h>
#include <algorithm>


namespace {

const QstateIdxType One = 1;
const QstateIdxType Two = 2;


template<class R>
struct abs2 {
    __device__ __forceinline__
    R operator()(const cuda_runtime::DeviceComplex &c) const {
        return c.re * c.re + c.im * c.im;
    }
};


struct null {
    __device__ __forceinline__
    const cuda_runtime::DeviceComplex &operator()(const cuda_runtime::DeviceComplex &c) const {
        return c;
    }
};


}


namespace cuda_runtime {

void DeviceQubitStates::allocate(const IdList &qregIdList) {
    deallocate();

    nQregIds_ = (int)qregIdList.size();
    QstateIdxType nStates = One << nQregIds_;
    size_t qregIdListSize = sizeof(int) * nQregIds_;
    throwOnError(cudaMalloc(&d_qregIdList_, qregIdListSize));
    throwOnError(cudaMemcpy(d_qregIdList_, qregIdList.data(), qregIdListSize, cudaMemcpyDefault));
    throwOnError(cudaMalloc(&d_qstates_, sizeof(Complex) * nStates));
}

void DeviceQubitStates::deallocate() {
    if (d_qregIdList_ != NULL)
        throwOnError(cudaFree(d_qregIdList_));
    if (d_qstates_ != NULL)
        throwOnError(cudaFree(d_qstates_));
    d_qregIdList_ = NULL;
    d_qstates_ = NULL;
}

void DeviceQubitStates::reset() {
    throwOnError(cudaMemset(d_qstates_, 0, sizeof(DeviceComplex) * nStates_));
    DeviceComplex cOne(1.);
    throwOnError(cudaMemcpy(d_qstates_, &cOne, sizeof(DeviceComplex), cudaMemcpyDefault));
}

__host__
QstateIdxType DeviceQubitStates::getNStates() const {
    return One << nQregIds_;
}

/* CUDAQubits */

CUDAQubits::CUDAQubits() {
    d_devQubitStatesArray_ = NULL;
}


CUDAQubits::~CUDAQubits() {
}

void CUDAQubits::addQubitStates(int key, CUDAQubitStates *qstates) {
    cuQubitStatesMap_[key] = qstates;
}

void CUDAQubits::detachQubitStates() {
    cuQubitStatesMap_.clear();
    freeDeviceBuffer();
}

void CUDAQubits::freeDeviceBuffer() {
    if (d_devQubitStatesArray_ != NULL)
        throwOnError(cudaFree(d_devQubitStatesArray_));
    d_devQubitStatesArray_ = NULL;
}

void CUDAQubits::prepare() {
    freeDeviceBuffer();
    
    size_t nQubitStates = cuQubitStatesMap_.size();
    DeviceQubitStates *dQstates = new DeviceQubitStates[nQubitStates];
    CUDAQubitStatesMap::iterator it = cuQubitStatesMap_.begin();
    for (int qstatesIdx = 0; (int)qstatesIdx < nQubitStates; ++qstatesIdx) {
        CUDAQubitStates &cuQstates = *it->second;
        dQstates[qstatesIdx] = cuQstates.getDeviceQubitStates();
        ++it;
    }

    size_t size = sizeof(DeviceQubitStates) * nQubitStates;
    throwOnError(cudaMalloc(&d_devQubitStatesArray_, size));
    throwOnError(cudaMemcpy(d_devQubitStatesArray_, dQstates, size, cudaMemcpyDefault));

    delete[] dQstates;
}
    
CUDAQubitStates &CUDAQubits::operator[](int key) {
    CUDAQubitStatesMap::iterator it = cuQubitStatesMap_.find(key);
    assert(it != cuQubitStatesMap_.end());
    return *it->second;
}

const CUDAQubitStates &CUDAQubits::operator[](int key) const {
    CUDAQubitStatesMap::const_iterator it = cuQubitStatesMap_.find(key);
    assert(it != cuQubitStatesMap_.end());
    return *it->second;
}

QstateIdxType CUDAQubits::getNStates() const {
    return One << cuQubitStatesMap_.size();
}


template<class V, class F>
void CUDAQubits::getValues(V *values,
                           QstateIdxType beginIdx, QstateIdxType endIdx,
                           const F &func, CUDARuntimeResource &rsrc) const {
    size_t nQubitStates = cuQubitStatesMap_.size();
    const DeviceQubitStates *d_devQubitStatesArray = d_devQubitStatesArray_;

    QstateIdxType stride = rsrc.hostMemSize<V>();
    V *h_values = rsrc.getHostMem<V>();

    /* FIXME: pipeline */

    for (QstateIdxType strideBegin = beginIdx; strideBegin < endIdx; strideBegin += stride) {
        QstateIdxType strideEnd = std::min(strideBegin + stride, endIdx);
        
        transform(strideBegin, strideEnd,
                  [=]__device__(QstateIdxType globalIdx) {                 
                      V v = real(1.);
                      for (int qstatesIdx = 0; qstatesIdx < nQubitStates; ++qstatesIdx) {
                          const DeviceQubitStates &dQstates = d_devQubitStatesArray[qstatesIdx];
                          /* getStateByGlobalIdx() */
                          QstateIdxType localIdx = 0;
                          for (int bitPos = 0; bitPos < dQstates.nQregIds_; ++bitPos) {
                              int qregId = dQstates.d_qregIdList_[bitPos]; 
                              if ((One << qregId) & globalIdx)
                                  localIdx |= One << bitPos;
                          }
                          const DeviceComplex &state = dQstates.d_qstates_[localIdx];
                          v *= func(state);
                      }
                      h_values[globalIdx - strideBegin] = v;
                  });
        throwOnError(cudaDeviceSynchronize());
        memcpy(&values[strideBegin], h_values, sizeof(V) * (strideEnd - strideBegin));
    }
}



void CUDAQubits::getStates(Complex *states,
                           QstateIdxType beginIdx, QstateIdxType endIdx,
                           CUDARuntimeResource &rsrc) const {

    getValues((DeviceComplex*)states, beginIdx, endIdx, null(), rsrc);
}

void CUDAQubits::getProbabilities(real *probArray,
                                  QstateIdxType beginIdx, QstateIdxType endIdx,
                                  CUDARuntimeResource &rsrc) const {
    
    getValues(probArray, beginIdx, endIdx, abs2<real>(), rsrc);
}



CUDAQubitStates::CUDAQubitStates() {
    
}

CUDAQubitStates::~CUDAQubitStates() {
    deallocate();
}



void CUDAQubitStates::allocate(const IdList &qregIdList) {
    
    qregIdList_ = qregIdList;
    devQstates_.allocate(qregIdList);
    devQstates_.reset();
}
    
void CUDAQubitStates::deallocate() {
    devQstates_.deallocate();
}

void CUDAQubitStates::reset() {
    devQstates_.reset();
}

int CUDAQubitStates::getLane(int qregId) const {
    IdList::const_iterator it = std::find(qregIdList_.begin(), qregIdList_.end(), qregId);
    assert(it != qregIdList_.end());
    return (int)std::distance(qregIdList_.begin(), it);
}


int cudaMeasure(real randNum, CUDAQubitStates &cuQstates, int qregId, CUDARuntimeResource &rsrc) {
    int cregValue = -1;
    
    int lane = cuQstates.getLane(qregId);

    QstateIdxType bitmask_lane = One << lane;
    QstateIdxType bitmask_hi = ~((Two << lane) - 1);
    QstateIdxType bitmask_lo = (One << lane) - 1;
    QstateIdxType nStates = One << (cuQstates.getNLanes() - 1);
    real prob = real(0.);

    DeviceComplex *d_qstates = cuQstates.getDevicePtr();
    prob = rsrc.deviceSum_(0, nStates,
                           [=] __device__(QstateIdxType idx) {
                               QstateIdxType idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                               const DeviceComplex &qs = d_qstates[idx_lo];
                               return abs2<real>()(qs);
                           });

    if (randNum < prob) {
        cregValue = 0;
        real norm = real(1.) / std::sqrt(prob);

        transform(0, nStates,
                  [=]__device__(QstateIdxType idx) {
                      QstateIdxType idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                      QstateIdxType idx_hi = idx_lo | bitmask_lane;
                      d_qstates[idx_lo] *= norm;
                      d_qstates[idx_hi] = real(0.);
                  });
    }
    else {
        cregValue = 1;
        real norm = real(1.) / std::sqrt(real(1.) - prob);
        transform(0, nStates,
                  [=]__device__(QstateIdxType idx) {
                      QstateIdxType idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                      QstateIdxType idx_hi = idx_lo | bitmask_lane;
                      d_qstates[idx_lo] = real(0.);
                      d_qstates[idx_hi] *= norm;
                  });
        
    }

    return cregValue;
}
    
void cudaApplyReset(CUDAQubitStates &cuQstates, int qregId) {
    int lane = cuQstates.getLane(qregId);

    QstateIdxType bitmask_lane = One << lane;
    QstateIdxType bitmask_hi = ~((Two << lane) - 1);
    QstateIdxType bitmask_lo = (One << lane) - 1;
    QstateIdxType nStates = One << (cuQstates.getNLanes() - 1);

    /* Assuming reset is able to be applyed after measurement.
     * Ref: https://quantumcomputing.stackexchange.com/questions/3908/possibility-of-a-reset-quantum-gate */
    DeviceComplex *d_qstates = cuQstates.getDevicePtr();
    transform(0, nStates,
              [=]__device__(QstateIdxType idx) {
                  QstateIdxType idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                  QstateIdxType idx_hi = idx_lo | bitmask_lane;
                  d_qstates[idx_lo] = d_qstates[idx_hi];
                  d_qstates[idx_hi] = real(0.);
              });
}

void cudaApplyUnaryGate(const CMatrix2x2 &mat, CUDAQubitStates &cuQstates, int qregId) {
    DeviceCMatrix2x2 dmat(mat);
    
    int lane = cuQstates.getLane(qregId);
    
    QstateIdxType bitmask_lane = One << lane;
    QstateIdxType bitmask_hi = ~((Two << lane) - 1);
    QstateIdxType bitmask_lo = (One << lane) - 1;
    QstateIdxType nStates = One << (cuQstates.getNLanes() - 1);

    DeviceComplex *d_qstates = cuQstates.getDevicePtr();
    transform(0, nStates,
              [=]__device__(QstateIdxType idx) {
                  QstateIdxType idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                  QstateIdxType idx_hi = idx_lo | bitmask_lane;
                  const DeviceComplex &qs0 = d_qstates[idx_lo];
                  const DeviceComplex &qs1 = d_qstates[idx_hi];
                  DeviceComplex qsout0 = dmat(0, 0) * qs0 + dmat(0, 1) * qs1;
                  DeviceComplex qsout1 = dmat(1, 0) * qs0 + dmat(1, 1) * qs1;
                  d_qstates[idx_lo] = qsout0;
                  d_qstates[idx_hi] = qsout1;
              });
}

void cudaApplyControlGate(const CMatrix2x2 &mat, CUDAQubitStates &cuQstates,
                          int controlId, int targetId) {
    int lane0 = cuQstates.getLane(controlId);
    int lane1 = cuQstates.getLane(targetId);
    QstateIdxType bitmask_control = One << lane0;
    QstateIdxType bitmask_target = One << lane1;
        
    QstateIdxType bitmask_lane_max = std::max(bitmask_control, bitmask_target);
    QstateIdxType bitmask_lane_min = std::min(bitmask_control, bitmask_target);
        
    QstateIdxType bitmask_hi = ~(bitmask_lane_max * 2 - 1);
    QstateIdxType bitmask_mid = (bitmask_lane_max - 1) & ~((bitmask_lane_min << 1) - 1);
    QstateIdxType bitmask_lo = bitmask_lane_min - 1;

    DeviceCMatrix2x2 dmat(mat);
    QstateIdxType nStates = One << (cuQstates.getNLanes() - 2);
    DeviceComplex *d_qstates = cuQstates.getDevicePtr();
    transform(0, nStates,
              [=]__device__(QstateIdxType idx) {
                  QstateIdxType idx_0 = ((idx << 2) & bitmask_hi) | ((idx << 1) & bitmask_mid) | (idx & bitmask_lo) | bitmask_control;
                  QstateIdxType idx_1 = idx_0 | bitmask_target;
                  
                  const DeviceComplex &qs0 = d_qstates[idx_0];
                  const DeviceComplex &qs1 = d_qstates[idx_1];;
                  DeviceComplex qsout0 = dmat(0, 0) * qs0 + dmat(0, 1) * qs1;
                  DeviceComplex qsout1 = dmat(1, 0) * qs0 + dmat(1, 1) * qs1;
                  d_qstates[idx_0] = qsout0;
                  d_qstates[idx_1] = qsout1;
              });
}

}
