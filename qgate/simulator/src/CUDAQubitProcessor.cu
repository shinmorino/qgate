#include "CUDAQubitProcessor.h"
#include "DeviceTypes.h"
#include "parallel.h"
#include "DeviceParallel.h"
#include "CUDAResource.h"
#include <algorithm>

using namespace qgate_cuda;

namespace {

    template<class R>
    struct abs2 {
        __device__ __forceinline__
            R operator()(const DeviceComplexType<R> &c) const {
            return c.real * c.real + c.imag * c.imag;
        }
    };

    template<class V>
    struct null {
        __device__ __forceinline__
            const DeviceComplexType<V> &operator()(const DeviceComplexType<V> &c) const {
            return c;
        }
    };

    template<class V> struct DeviceType;
    template<> struct DeviceType<float> { typedef float Type; };
    template<> struct DeviceType<double> { typedef double Type; };
    template<> struct DeviceType<qgate::ComplexType<float>> { typedef DeviceComplexType<float> Type; };
    template<> struct DeviceType<qgate::ComplexType<double>> { typedef DeviceComplexType<double> Type; };
}


using qgate::Qone;
using qgate::Qtwo;


template<class real>
CUDAQubitProcessor<real>::CUDAQubitProcessor(CUDAResource &rsrc) : rsrc_(rsrc) { }

template<class real>
CUDAQubitProcessor<real>::~CUDAQubitProcessor() { }

template<class real>
void CUDAQubitProcessor<real>::prepare(qgate::QubitStates &qstates) {
    deviceSum_.prepare();
}

template<class real>
int CUDAQubitProcessor<real>::measure(double randNum, qgate::QubitStates &qstates, int qregId) const {

    CUDAQubitStates<real> &cuQstates = static_cast<CUDAQubitStates<real>&>(qstates);
    
    int cregValue = -1;

    int lane = cuQstates.getLane(qregId);

    QstateIdx bitmask_lane = Qone << lane;
    QstateIdx bitmask_hi = ~((Qtwo << lane) - 1);
    QstateIdx bitmask_lo = (Qone << lane) - 1;
    QstateIdx nStates = Qone << (cuQstates.getNQregs() - 1);
    real prob = real(0.);

    DeviceComplex *d_qstates = cuQstates.getDevicePtr();
    deviceSum_.allocate(rsrc_);
    prob = deviceSum_(0, nStates,
                      [=] __device__(QstateIdx idx) {
                          QstateIdx idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                          const DeviceComplex &qs = d_qstates[idx_lo];
                          return abs2<real>()(qs);
                          });
    deviceSum_.deallocate(rsrc_);

    if (real(randNum) < prob) {
        cregValue = 0;
        real norm = real(1.) / std::sqrt(prob);

        transform(0, nStates,
                  [=]__device__(QstateIdx idx) {
                      QstateIdx idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                      QstateIdx idx_hi = idx_lo | bitmask_lane;
                      d_qstates[idx_lo] *= norm;
                      d_qstates[idx_hi] = real(0.);
                  });
    }
    else {
        cregValue = 1;
        real norm = real(1.) / std::sqrt(real(1.) - prob);
        transform(0, nStates,
                  [=]__device__(QstateIdx idx) {
                      QstateIdx idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                      QstateIdx idx_hi = idx_lo | bitmask_lane;
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

    QstateIdx bitmask_lane = Qone << lane;
    QstateIdx bitmask_hi = ~((Qtwo << lane) - 1);
    QstateIdx bitmask_lo = (Qone << lane) - 1;
    QstateIdx nStates = Qone << (cuQstates.getNQregs() - 1);

    /* Assuming reset is able to be applyed after measurement.
     * Ref: https://quantumcomputing.stackexchange.com/questions/3908/possibility-of-a-reset-quantum-gate */
    DeviceComplex *d_qstates = cuQstates.getDevicePtr();
    transform(0, nStates,
              [=]__device__(QstateIdx idx) {
                  QstateIdx idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                  QstateIdx idx_hi = idx_lo | bitmask_lane;
                  d_qstates[idx_lo] = d_qstates[idx_hi];
                  d_qstates[idx_hi] = real(0.);
              });
}

template<class real>
void CUDAQubitProcessor<real>::applyUnaryGate(const Matrix2x2C64 &mat, qgate::QubitStates &qstates, int qregId) const {
    CUDAQubitStates<real> &cuQstates = static_cast<CUDAQubitStates<real>&>(qstates);

    DeviceMatrix2x2C<real> dmat(mat);
    
    int lane = cuQstates.getLane(qregId);
    
    QstateIdx bitmask_lane = Qone << lane;
    QstateIdx bitmask_hi = ~((Qtwo << lane) - 1);
    QstateIdx bitmask_lo = (Qone << lane) - 1;
    QstateIdx nStates = Qone << (cuQstates.getNQregs() - 1);

    DeviceComplex *d_qstates = cuQstates.getDevicePtr();
    transform(0, nStates,
              [=]__device__(QstateIdx idx) {
                  typedef DeviceComplexType<real> DeviceComplex;
                  QstateIdx idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                  QstateIdx idx_hi = idx_lo | bitmask_lane;
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
    QstateIdx bitmask_control = Qone << lane0;
    QstateIdx bitmask_target = Qone << lane1;
        
    QstateIdx bitmask_lane_max = std::max(bitmask_control, bitmask_target);
    QstateIdx bitmask_lane_min = std::min(bitmask_control, bitmask_target);
        
    QstateIdx bitmask_hi = ~(bitmask_lane_max * 2 - 1);
    QstateIdx bitmask_mid = (bitmask_lane_max - 1) & ~((bitmask_lane_min << 1) - 1);
    QstateIdx bitmask_lo = bitmask_lane_min - 1;
    
    DeviceMatrix2x2C<real> dmat(mat);
    QstateIdx nStates = Qone << (cuQstates.getNQregs() - 2);
    DeviceComplex *d_qstates = cuQstates.getDevicePtr();
    transform(0, nStates,
              [=]__device__(QstateIdx idx) {
                  QstateIdx idx_0 = ((idx << 2) & bitmask_hi) | ((idx << 1) & bitmask_mid) | (idx & bitmask_lo) | bitmask_control;
                  QstateIdx idx_1 = idx_0 | bitmask_target;
                  
                  const DeviceComplex &qs0 = d_qstates[idx_0];
                  const DeviceComplex &qs1 = d_qstates[idx_1];;
                  DeviceComplex qsout0 = dmat(0, 0) * qs0 + dmat(0, 1) * qs1;
                  DeviceComplex qsout1 = dmat(1, 0) * qs0 + dmat(1, 1) * qs1;
                  d_qstates[idx_0] = qsout0;
                  d_qstates[idx_1] = qsout1;
              });
}



template<class real> template<class R, class F>
void CUDAQubitProcessor<real>::getStates(R *values, const F &func,
                                         const QubitStatesList &qstatesList,
                                         QstateIdx beginIdx, QstateIdx endIdx) const {
    int nQubitStates = (int)qstatesList.size();
    DeviceQubitStates<real> *d_devQubitStatesArray = rsrc_.getDeviceMem<DeviceQubitStates<real>>(nQubitStates);

    typedef typename DeviceType<R>::Type DeviceR;

    QstateIdx stride = rsrc_.hostMemSize<R>();
    DeviceR *h_values = rsrc_.getHostMem<DeviceR>(stride);

    DeviceQubitStates<real> *dQstates = new DeviceQubitStates<real>[nQubitStates];
    for (int idx = 0; idx < nQubitStates; ++idx) {
        const CUDAQubitStates<real> &cuQstates = *static_cast<CUDAQubitStates<real>*>(qstatesList[idx]);
        dQstates[idx] = cuQstates.getDeviceQubitStates();
    }

    size_t size = sizeof(DeviceQubitStates<real>) * nQubitStates;
    throwOnError(cudaMemcpyAsync(d_devQubitStatesArray, dQstates, size, cudaMemcpyDefault));
    
    /* FIXME: pipeline */

    for (QstateIdx strideBegin = beginIdx; strideBegin < endIdx; strideBegin += stride) {
        QstateIdx strideEnd = std::min(strideBegin + stride, endIdx);
        
        transform(strideBegin, strideEnd,
                  [=]__device__(QstateIdx globalIdx) {                 
                      DeviceR v = DeviceR(1.);
                      for (int qstatesIdx = 0; qstatesIdx < nQubitStates; ++qstatesIdx) {
                          const DeviceQubitStates<real> &dQstates = d_devQubitStatesArray[qstatesIdx];
                          /* getStateByGlobalIdx() */
                          QstateIdx localIdx = 0;
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
        R *h_values_cpu = reinterpret_cast<R*>(h_values);
        qgate_cpu::parallel_for_each(strideBegin, strideEnd,
                                    [=](QstateIdx idx) {
                                        values[idx] *= h_values_cpu[idx - strideBegin];
                                    }
                                    );
    }
}


template<class real>
void CUDAQubitProcessor<real>::getStates(void *array, QstateIdx arrayOffset,
                                         MathOp op,
                                         const QubitStatesList &qstatesList,
                                         QstateIdx beginIdx, QstateIdx endIdx) const {

    const qgate::QubitStates *qstates = qstatesList[0];
    if (sizeof(real) == sizeof(float))
        abortIf(qstates->getPrec() != qgate::precFP32, "Wrong type");
    else if (sizeof(real) == sizeof(double))
        abortIf(qstates->getPrec() != qgate::precFP64, "Wrong type");

    switch (op) {
    case qgate::mathOpNull: {
        Complex *cmpArray = static_cast<Complex*>(array);
        getStates(&cmpArray[arrayOffset], null<real>(), qstatesList, beginIdx, endIdx);
        break;
    }
    case qgate::mathOpProb: {
        real *vArray = static_cast<real*>(array);
        getStates(&vArray[arrayOffset], abs2<real>(), qstatesList, beginIdx, endIdx);
        break;
    }
    default:
        abort_("Unknown math op.");
    }

}

template class CUDAQubitProcessor<float>;
template class CUDAQubitProcessor<double>;
