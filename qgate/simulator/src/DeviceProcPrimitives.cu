#include "DeviceProcPrimitives.h"
#include "DeviceParallel.h"
#include "parallel.h"
#include "DeviceSum.cuh"
#include "DeviceFunctors.cuh"
#include <algorithm>

using namespace qgate_cuda;
using qgate::QstateIdx;
using qgate::QstateSize;
using qgate::Qone;
using qgate::Qtwo;


template<class real>
DeviceProcPrimitives<real>::DeviceProcPrimitives(CUDADevice &device) : device_(device), deviceSum_(device) {
}

template<class real>
void DeviceProcPrimitives<real>::set(DevicePtrs &d_qStatesPtrs,
                                     const void *pv, QstateIdx offset, qgate::QstateSize size) {
    DeviceComplex *d_buf = d_qStatesPtrs.getPtr(offset);
    device_.makeCurrent();
    throwOnError(cudaMemcpyAsync(d_buf, pv, size, cudaMemcpyDefault));
}

template<class real>
void DeviceProcPrimitives<real>::fillZero(DevicePtrs &d_qStatesPtrs,
                                          qgate::QstateIdx begin, qgate::QstateIdx end) {
    DeviceComplex *d_buf = d_qStatesPtrs.getPtr(begin);
    QstateSize size = end - begin;

    device_.makeCurrent();
    throwOnError(cudaMemsetAsync(d_buf, 0, sizeof(DeviceComplex) * size));
}


template<class real>
void DeviceProcPrimitives<real>::traceOut_launch(DevicePtrs &d_qStatesPtrs, int lane,
                                                 qgate::QstateIdx begin, qgate::QstateIdx end) {
    QstateIdx bit = Qone << lane;
    QstateIdx bitmask_hi = ~((bit << 1) - 1);
    QstateIdx bitmask_lo = bit - 1;
    
    device_.makeCurrent();
    deviceSum_.launch(begin, end, [=] __device__(QstateIdx idx) {
                QstateIdx idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                return abs2<real>()(d_qStatesPtrs[idx_lo]);
            });
}
    
template<class real>
real DeviceProcPrimitives<real>::traceOut_sync() {
    device_.makeCurrent();
    return deviceSum_.sync();
}
    
template<class real>
void DeviceProcPrimitives<real>::measure_set0(DevicePtrs &d_qStatesPtrs, int lane, real prob,
                                              qgate::QstateIdx begin, qgate::QstateIdx end) {
    
    QstateIdx nThreads = end - begin;

    QstateIdx bitmask_lane = Qone << lane;
    QstateIdx bitmask_hi = ~((Qtwo << lane) - 1);
    QstateIdx bitmask_lo = (Qone << lane) - 1;

    device_.makeCurrent();
    real norm = real(1.) / std::sqrt(prob);
    transform(0, nThreads,
              [=]__device__(QstateIdx idx) mutable {
                  idx += begin;
                  QstateIdx idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                  QstateIdx idx_hi = idx_lo | bitmask_lane;
                  d_qStatesPtrs[idx_lo] *= norm;
                  d_qStatesPtrs[idx_hi] = real(0.);
              });
}
    
template<class real>
void DeviceProcPrimitives<real>::measure_set1(DevicePtrs &d_qStatesPtrs, int lane, real prob,
                                              qgate::QstateIdx begin, qgate::QstateIdx end) {
    QstateIdx nThreads = end - begin;

    QstateIdx bitmask_lane = Qone << lane;
    QstateIdx bitmask_hi = ~((Qtwo << lane) - 1);
    QstateIdx bitmask_lo = (Qone << lane) - 1;

    device_.makeCurrent();
    real norm = real(1.) / std::sqrt(real(1.) - prob);
    transform(0, nThreads,
              [=]__device__(QstateIdx idx) mutable {
                  idx += begin;
                  QstateIdx idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                  QstateIdx idx_hi = idx_lo | bitmask_lane;
                  d_qStatesPtrs[idx_lo] = real(0.);
                  d_qStatesPtrs[idx_hi] *= norm;
              });
}
    
template<class real>
void DeviceProcPrimitives<real>::applyReset(DevicePtrs &d_qStatesPtrs, int lane,
                                            qgate::QstateIdx begin, qgate::QstateIdx end) {
    QstateIdx nThreads = end - begin;

    QstateIdx bitmask_lane = Qone << lane;
    QstateIdx bitmask_hi = ~((Qtwo << lane) - 1);
    QstateIdx bitmask_lo = (Qone << lane) - 1;

    /* Assuming reset is able to be applyed after measurement.
     * Ref: https://quantumcomputing.stackexchange.com/questions/3908/possibility-of-a-reset-quantum-gate */
    device_.makeCurrent();
    transform(0, nThreads,
              [=]__device__(QstateIdx idx) mutable {
                  idx += begin;
                  QstateIdx idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                  QstateIdx idx_hi = idx_lo | bitmask_lane;
                  d_qStatesPtrs[idx_lo] = d_qStatesPtrs[idx_hi];
                  d_qStatesPtrs[idx_hi] = real(0.);
              });
}

template<class real>
void DeviceProcPrimitives<real>::applyUnaryGate(const DeviceMatrix2x2C<real> &mat,
                                                DevicePtrs &d_qStatesPtrs, int lane,
                                                qgate::QstateIdx begin, qgate::QstateIdx end) {
    DeviceMatrix2x2C<real> dmat(mat);

    QstateIdx nThreads = end - begin;
    
    QstateIdx bitmask_lane = Qone << lane;
    QstateIdx bitmask_hi = ~((Qtwo << lane) - 1);
    QstateIdx bitmask_lo = (Qone << lane) - 1;
    
    device_.makeCurrent();
    transform(0, nThreads,
              [=]__device__(QstateIdx idx) mutable {
                  typedef DeviceComplexType<real> DeviceComplex;
                  QstateIdx idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                  QstateIdx idx_hi = idx_lo | bitmask_lane;
                  const DeviceComplex &qs0 = d_qStatesPtrs[idx_lo];
                  const DeviceComplex &qs1 = d_qStatesPtrs[idx_hi];
                  DeviceComplex qsout0 = dmat(0, 0) * qs0 + dmat(0, 1) * qs1;
                  DeviceComplex qsout1 = dmat(1, 0) * qs0 + dmat(1, 1) * qs1;
                  d_qStatesPtrs[idx_lo] = qsout0;
                  d_qStatesPtrs[idx_hi] = qsout1;
              });    
}

template<class real> void DeviceProcPrimitives<real>::
applyControlGate(const DeviceMatrix2x2C<real> &mat,
                 DevicePtrs &d_qStatesPtrs, int controlLane, int targetLane,
                 qgate::QstateIdx begin, qgate::QstateIdx end) {

    QstateIdx nThreads = end - begin;

    QstateIdx bitmask_control = Qone << controlLane;
    QstateIdx bitmask_target = Qone << targetLane;
        
    QstateIdx bitmask_lane_max = std::max(bitmask_control, bitmask_target);
    QstateIdx bitmask_lane_min = std::min(bitmask_control, bitmask_target);
        
    QstateIdx bitmask_hi = ~(bitmask_lane_max * 2 - 1);
    QstateIdx bitmask_mid = (bitmask_lane_max - 1) & ~((bitmask_lane_min << 1) - 1);
    QstateIdx bitmask_lo = bitmask_lane_min - 1;
    
    DeviceMatrix2x2C<real> dmat(mat);
    device_.makeCurrent();
    transform(0, nThreads,
              [=]__device__(QstateIdx idx) mutable {
                  idx += begin;
                  QstateIdx idx_0 = ((idx << 2) & bitmask_hi) | ((idx << 1) & bitmask_mid) | (idx & bitmask_lo) | bitmask_control;
                  QstateIdx idx_1 = idx_0 | bitmask_target;
                  
                  const DeviceComplex &qs0 = d_qStatesPtrs[idx_0];
                  const DeviceComplex &qs1 = d_qStatesPtrs[idx_1];;
                  DeviceComplex qsout0 = dmat(0, 0) * qs0 + dmat(0, 1) * qs1;
                  DeviceComplex qsout1 = dmat(1, 0) * qs0 + dmat(1, 1) * qs1;
                  d_qStatesPtrs[idx_0] = qsout0;
                  d_qStatesPtrs[idx_1] = qsout1;
              });
    
}

template class DeviceProcPrimitives<float>;
template class DeviceProcPrimitives<double>;
