#include "DeviceProcPrimitives.h"
#include "MultiDevicePtr.cuh"
#include "DeviceParallel.h"
#include "parallel.h"

using namespace qgate_cuda;
using qgate::QstateIdx;
using qgate::QstateSize;
using qgate::Qone;
using qgate::Qtwo;


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


template<class real>
DeviceProcPrimitives<real>::DeviceProcPrimitives(CUDADevice &device) : device_(device), deviceSum_(device) {
}

template<class real>
void DeviceProcPrimitives<real>::set(DevQubitStates &devQstates,
                                     const void *pv, QstateIdx offset, qgate::QstateSize size) {
    MultiDevicePtr<DeviceComplex> d_qstates(devQstates.d_qStatesPtrs, devQstates.nLanesInDevice);
    DeviceComplex *d_buf = d_qstates.getPtr(offset);
    device_.makeCurrent();
    DeviceComplex cOne(1.);
    throwOnError(cudaMemcpyAsync(d_buf, &cOne, sizeof(DeviceComplex), cudaMemcpyDefault));
}

template<class real>
void DeviceProcPrimitives<real>::fillZero(DevQubitStates &devQstates,
                                          qgate::QstateIdx begin, qgate::QstateIdx end) {
    MultiDevicePtr<DeviceComplex> d_qstates(devQstates.d_qStatesPtrs, devQstates.nLanesInDevice);
    DeviceComplex *d_buf = d_qstates.getPtr(begin);
    QstateSize size = end - begin;

    device_.makeCurrent();
    throwOnError(cudaMemsetAsync(d_buf, 0, sizeof(DeviceComplex) * size));
}


template<class real>
void DeviceProcPrimitives<real>::traceOut_launch(DevQubitStates &devQstates, int lane,
                                                 qgate::QstateIdx begin, qgate::QstateIdx end) {
    MultiDevicePtr<DeviceComplex> d_qstates(devQstates.d_qStatesPtrs, devQstates.nLanesInDevice); 

    QstateIdx bit = Qone << lane;
    QstateIdx bitmask_hi = ~((bit << 1) - 1);
    QstateIdx bitmask_lo = bit - 1;
    
    device_.makeCurrent();
    deviceSum_.launch(begin, end, [=] __device__(QstateIdx idx) {
                QstateIdx idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                return abs2<real>()(d_qstates[idx_lo]);
            });
}
    
template<class real>
real DeviceProcPrimitives<real>::traceOut_sync() {
    device_.makeCurrent();
    return deviceSum_.sync();
}
    
template<class real>
void DeviceProcPrimitives<real>::measure_set0(DevQubitStates &devQstates, int lane, real prob,
                                              qgate::QstateIdx begin, qgate::QstateIdx end) {
    
    QstateIdx nThreads = end - begin;
    MultiDevicePtr<DeviceComplex> d_qstates(devQstates.d_qStatesPtrs, devQstates.nLanesInDevice); 

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
                  d_qstates[idx_lo] *= norm;
                  d_qstates[idx_hi] = real(0.);
              });
}
    
template<class real>
void DeviceProcPrimitives<real>::measure_set1(DevQubitStates &devQstates, int lane, real prob,
                                              qgate::QstateIdx begin, qgate::QstateIdx end) {
    QstateIdx nThreads = end - begin;
    MultiDevicePtr<DeviceComplex> d_qstates(devQstates.d_qStatesPtrs, devQstates.nLanesInDevice); 

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
                  d_qstates[idx_lo] = real(0.);
                  d_qstates[idx_hi] *= norm;
              });
}
    
template<class real>
void DeviceProcPrimitives<real>::applyReset(DevQubitStates &devQstates, int lane,
                                            qgate::QstateIdx begin, qgate::QstateIdx end) {
    QstateIdx nThreads = end - begin;
    MultiDevicePtr<DeviceComplex> d_qstates(devQstates.d_qStatesPtrs, devQstates.nLanesInDevice); 

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
                  d_qstates[idx_lo] = d_qstates[idx_hi];
                  d_qstates[idx_hi] = real(0.);
              });
}

template<class real>
void DeviceProcPrimitives<real>::applyUnaryGate(const DeviceMatrix2x2C<real> &mat,
                                                DevQubitStates &devQstates, int lane,
                                                qgate::QstateIdx begin, qgate::QstateIdx end) {
    DeviceMatrix2x2C<real> dmat(mat);

    QstateIdx nThreads = end - begin;
    MultiDevicePtr<DeviceComplex> d_qstates(devQstates.d_qStatesPtrs, devQstates.nLanesInDevice); 
    
    QstateIdx bitmask_lane = Qone << lane;
    QstateIdx bitmask_hi = ~((Qtwo << lane) - 1);
    QstateIdx bitmask_lo = (Qone << lane) - 1;
    
    device_.makeCurrent();
    transform(0, nThreads,
              [=]__device__(QstateIdx idx) mutable {
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

template<class real> void DeviceProcPrimitives<real>::
applyControlGate(const DeviceMatrix2x2C<real> &mat,
                 DevQubitStates &devQstates, int controlLane, int targetLane,
                 qgate::QstateIdx begin, qgate::QstateIdx end) {

    QstateIdx nThreads = end - begin;
    MultiDevicePtr<DeviceComplex> d_qstates(devQstates.d_qStatesPtrs, devQstates.nLanesInDevice); 

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
                  
                  const DeviceComplex &qs0 = d_qstates[idx_0];
                  const DeviceComplex &qs1 = d_qstates[idx_1];;
                  DeviceComplex qsout0 = dmat(0, 0) * qs0 + dmat(0, 1) * qs1;
                  DeviceComplex qsout1 = dmat(1, 0) * qs0 + dmat(1, 1) * qs1;
                  d_qstates[idx_0] = qsout0;
                  d_qstates[idx_1] = qsout1;
              });
    
}

template<class real>
void DeviceProcPrimitives<real>::
getStates(void *array, qgate::QstateIdx arrayOffset, qgate::MathOp op,
          const DevQubitStates &devQstates,
          qgate::QstateIdx begin, qgate::QstateIdx end) {

    switch (op) {
    case qgate::mathOpNull: {
        Complex *cmpArray = static_cast<Complex*>(array);
        getStates(&cmpArray[arrayOffset], null<real>(), devQstates, begin, end);
        break;
    }
    case qgate::mathOpProb: {
        real *vArray = static_cast<real*>(array);
        getStates(&vArray[arrayOffset], abs2<real>(), devQstates, begin, end);
        break;
    }
    default:
        abort_("Unknown math op.");
    }
}


template<class real> template<class R, class F>
void DeviceProcPrimitives<real>::getStates(R *values, const F &op,
                                           const DevQubitStates &devQstates,
                                           qgate::QstateIdx begin, qgate::QstateIdx end) {
    /* FIXME: multiple qstates cannot be handled effectively. */

    typedef typename DeviceType<R>::Type DeviceR;

    QstateIdx stride = device_.tmpHostMemSize<R>();
    DeviceR *h_values = device_.getTmpHostMem<DeviceR>(stride);
    
    /* FIXME: pipeline */

    device_.makeCurrent();
    for (QstateIdx strideBegin = begin; strideBegin < end; strideBegin += stride) {
        QstateIdx strideEnd = std::min(strideBegin + stride, end);
        MultiDevicePtr<DeviceComplex> d_qstates(devQstates.d_qStatesPtrs, devQstates.nLanesInDevice); 
        transform(strideBegin, strideEnd,
                  [=]__device__(QstateIdx globalIdx) {                 
                      DeviceR v = DeviceR(1.);
                      /* getStateByGlobalIdx() */
                      QstateIdx localIdx = 0;
                      for (int bitPos = 0; bitPos < devQstates.nLanes; ++bitPos) {
                          int qregId = devQstates.qregIdToLane[bitPos]; 
                          if ((Qone << qregId) & globalIdx)
                              localIdx |= Qone << bitPos;
                      }
                      const DeviceComplex &state = d_qstates[localIdx];
                      v *= op(state);
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


template class DeviceProcPrimitives<float>;
template class DeviceProcPrimitives<double>;
