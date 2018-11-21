#include "CUDAQubitProcessor.h"
#include "DeviceTypes.h"
#include "DeviceParallel.h"
#include "CUDAResource.h"


using namespace qgate_cuda;

namespace {

template<class R>
struct abs2 {
    __device__ __forceinline__
    R operator()(const DeviceComplexType<R> &c) const {
        return c.re * c.re + c.im * c.im;
    }
};

struct null {
    template<class V>
    __device__ __forceinline__
    const DeviceComplexType<V> &operator()(const DeviceComplexType<V> &c) const {
        return c;
    }
};

}


using qgate::Qone;
using qgate::Qtwo;


template<class real>
CUDAQubitProcessor<real>::CUDAQubitProcessor(CUDAResource &rsrc) : rsrc_(rsrc) { }
template<class real>
CUDAQubitProcessor<real>::~CUDAQubitProcessor() { }

template<class real>
void CUDAQubitProcessor<real>::prepare(qgate::QubitStates &qstates) {
}

template<class real>
int CUDAQubitProcessor<real>::measure(double randNum, qgate::QubitStates &qstates, int qregId) const {

    CUDAQubitStates<real> &cuQstates = static_cast<CUDAQubitStates<real>&>(qstates);
    
    int cregValue = -1;

    int lane = cuQstates.getLane(qregId);

    QstateIdxType bitmask_lane = Qone << lane;
    QstateIdxType bitmask_hi = ~((Qtwo << lane) - 1);
    QstateIdxType bitmask_lo = (Qone << lane) - 1;
    QstateIdxType nStates = Qone << (cuQstates.getNLanes() - 1);
    real prob = real(0.);

    DeviceComplex *d_qstates = cuQstates.getDevicePtr();
    prob = rsrc_.deviceSum_(0, nStates,
                            [=] __device__(QstateIdxType idx) {
                                QstateIdxType idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                                const DeviceComplex &qs = d_qstates[idx_lo];
                                return abs2<real>()(qs);
                            });
    
    if (real(randNum) < prob) {
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
    

template<class real>
void CUDAQubitProcessor<real>::applyReset(qgate::QubitStates &qstates, int qregId) const {
    CUDAQubitStates<real> &cuQstates = static_cast<CUDAQubitStates<real>&>(qstates);
    
    int lane = cuQstates.getLane(qregId);

    QstateIdxType bitmask_lane = Qone << lane;
    QstateIdxType bitmask_hi = ~((Qtwo << lane) - 1);
    QstateIdxType bitmask_lo = (Qone << lane) - 1;
    QstateIdxType nStates = Qone << (cuQstates.getNLanes() - 1);

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

template<class real>
void CUDAQubitProcessor<real>::applyUnaryGate(const Matrix2x2C64 &mat, qgate::QubitStates &qstates, int qregId) const {
    CUDAQubitStates<real> &cuQstates = static_cast<CUDAQubitStates<real>&>(qstates);

    DeviceCMatrix2x2 dmat(mat);
    
    int lane = cuQstates.getLane(qregId);
    
    QstateIdxType bitmask_lane = Qone << lane;
    QstateIdxType bitmask_hi = ~((Qtwo << lane) - 1);
    QstateIdxType bitmask_lo = (Qone << lane) - 1;
    QstateIdxType nStates = Qone << (cuQstates.getNLanes() - 1);

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

template<class real>
void CUDAQubitProcessor<real>::applyControlGate(const Matrix2x2C64 &mat, QubitStates &qstates,
                                                int controlId, int targetId) const {
    CUDAQubitStates<real> &cuQstates = static_cast<CUDAQubitStates<real>&>(qstates);
    
    int lane0 = cuQstates.getLane(controlId);
    int lane1 = cuQstates.getLane(targetId);
    QstateIdxType bitmask_control = Qone << lane0;
    QstateIdxType bitmask_target = Qone << lane1;
        
    QstateIdxType bitmask_lane_max = std::max(bitmask_control, bitmask_target);
    QstateIdxType bitmask_lane_min = std::min(bitmask_control, bitmask_target);
        
    QstateIdxType bitmask_hi = ~(bitmask_lane_max * 2 - 1);
    QstateIdxType bitmask_mid = (bitmask_lane_max - 1) & ~((bitmask_lane_min << 1) - 1);
    QstateIdxType bitmask_lo = bitmask_lane_min - 1;
    
    DeviceCMatrix2x2 dmat(mat);
    QstateIdxType nStates = Qone << (cuQstates.getNLanes() - 2);
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



template<class real> template<class F>
void CUDAQubitProcessor<real>::getValues(real *values, QstateIdxType arrayOffset,
                                         MathOp op,
                                         const QubitStatesList &qstatesList,
                                         QstateIdxType beginIdx, QstateIdxType endIdx,
                                         const F &func) const {
    size_t nQubitStates = qstatesList.size();
    const DeviceQubitStates<real> *d_devQubitStatesArray = rsrc_.getDeviceMem<DeviceQubitStates<real>>();

    QstateIdxType stride = rsrc_.hostMemSize<real>();
    real *h_values = rsrc_.getHostMem<real>();

    DeviceQubitStates<real> *dQstates = new DeviceQubitStates<real>[nQubitStates];
    for (int idx = 0; idx < nQubitStates; ++idx) {
        const CUDAQubitStates<real> &cuQstates = static_cast<CUDAQubitStates<real>&>(qstatesList[idx]);
        dQstates[idx] = cuQstates.getDeviceQubitStates();
    }

    size_t size = sizeof(DeviceQubitStates<real>) * nQubitStates;
    throwOnError(cudaMemcpy(d_devQubitStatesArray, dQstates, size, cudaMemcpyDefault));
    
    /* FIXME: pipeline */

    for (QstateIdxType strideBegin = beginIdx; strideBegin < endIdx; strideBegin += stride) {
        QstateIdxType strideEnd = std::min(strideBegin + stride, endIdx);
        
        transform(strideBegin, strideEnd,
                  [=]__device__(QstateIdxType globalIdx) {                 
                      real v = real(1.);
                      for (int qstatesIdx = 0; qstatesIdx < nQubitStates; ++qstatesIdx) {
                          const DeviceQubitStates<real> &dQstates = d_devQubitStatesArray[qstatesIdx];
                          /* getStateByGlobalIdx() */
                          QstateIdxType localIdx = 0;
                          for (int bitPos = 0; bitPos < dQstates.nQregIds_; ++bitPos) {
                              int qregId = dQstates.d_qregIdList_[bitPos]; 
                              if ((Qone << qregId) & globalIdx)
                                  localIdx |= Qone << bitPos;
                          }
                          const DeviceComplex &state = dQstates.d_qstates_[localIdx];
                          v *= func(state);
                      }
                      h_values[globalIdx - strideBegin] = v;
                  });
        throwOnError(cudaDeviceSynchronize());
        parallel_for(strideBegin, strideEnd,
                     [=](QstateIdxType idx) {
                         values[idx] *= h_values[idx - strideBegin];
                     }
                );
    }
}


template<class real>
void CUDAQubitProcessor<real>::getStates(void *array, QstateIdxType arrayOffset,
                                         MathOp op,
                                         const QubitStatesList &qstatesList,
                                         QstateIdxType beginIdx, QstateIdxType endIdx) const {
    
    
    
    
}

