#include "DeviceProcPrimitives.h"
#include "DeviceParallel.h"
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
void DeviceProcPrimitives<real>::calcProb_launch(const DevicePtrs &d_qStatesPtrs, int lane,
                                                 qgate::QstateIdx begin, qgate::QstateIdx end) {
    QstateIdx lane_bit = Qone << lane;
    QstateIdx bitmask_hi = ~((lane_bit << 1) - 1);
    QstateIdx bitmask_lo = lane_bit - 1;
    
    device_.makeCurrent();
    deviceSum_.launch(begin, end, [=] __device__(QstateIdx idx) {
                QstateIdx idx_0 = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                return abs2<real>()(d_qStatesPtrs[idx_0]);
            });
}
    
template<class real>
real DeviceProcPrimitives<real>::calcProb_sync() {
    device_.makeCurrent();
    return deviceSum_.sync();
}

template<class real>
void DeviceProcPrimitives<real>::copyAndFillZero(DevicePtrs &dstPtrs,
                                                 const DevicePtrs &srcPtrs, qgate::QstateIdx srcSize,
                                                 qgate::QstateIdx begin, qgate::QstateIdx end) {
    device_.makeCurrent();
    transform(begin, end,
              [=]__device__(QstateIdx idx) mutable {
        DeviceComplex vSrc = (idx < srcSize) ? srcPtrs[idx] : DeviceComplex(0.);
        dstPtrs[idx] = vSrc;
    });
}

/* FIXME: optimize */
template<class real>
void DeviceProcPrimitives<real>::kron(DevicePtrs &dstPtrs,
                                      const DevicePtrs &srcPtrs0, qgate::QstateSize Nsrc0,
                                      const DevicePtrs &srcPtrs1, qgate::QstateSize Nsrc1,
                                      qgate::QstateIdx begin, qgate::QstateIdx end) {
    device_.makeCurrent();
    transform(begin, end,
              [=]__device__(QstateIdx dstIdx) mutable {
                  QstateIdx idx0 = dstIdx / Nsrc1;
                  QstateIdx idx1 = dstIdx % Nsrc1;
                  dstPtrs[dstIdx] = srcPtrs0[idx0] * srcPtrs1[idx1];
              });
}

/* FIXME: optimize */
template<class real>
void DeviceProcPrimitives<real>::kronInPlace_0(DevicePtrs &dstPtrs, qgate::QstateSize Ndst,
                                               const DevicePtrs &srcPtrs, qgate::QstateSize Nsrc,
                                               qgate::QstateIdx begin, qgate::QstateIdx end) {
    device_.makeCurrent();
    transform(begin, end,
              [=]__device__(QstateIdx dstIdx) mutable {
                  if (Ndst <= dstIdx) {
                      QstateIdx dstInIdx = dstIdx % Ndst;
                      QstateIdx srcIdx = dstIdx / Ndst;
                      dstPtrs[dstIdx] = dstPtrs[dstInIdx] * srcPtrs[srcIdx];
                  }
              });
}

template<class real>
void DeviceProcPrimitives<real>::kronInPlace_1(DevicePtrs &dstPtrs, qgate::QstateSize Ndst,
                                               const DevicePtrs &srcPtrs, qgate::QstateSize Nsrc,
                                               qgate::QstateIdx begin, qgate::QstateIdx end) {
    device_.makeCurrent();
    transform(begin, end,
              [=]__device__(QstateIdx dstIdx) mutable {
                  if (dstIdx < Ndst)
                      dstPtrs[dstIdx] *= srcPtrs[0];
              });
}


template<class real>
void DeviceProcPrimitives<real>::decohere(DevicePtrs &d_qStatesPtrs,
                                          int lane, int value, real prob,
                                          qgate::QstateIdx begin, qgate::QstateIdx end) {
    QstateIdx lane_bit = Qone << lane;

    device_.makeCurrent();
    if (value == 0) {
        real norm = real(1.) / std::sqrt(prob);
        transform(begin, end,
                  [=]__device__(QstateIdx idx) mutable {
                      DeviceComplex v;
                      if ((idx & lane_bit) == 0)
                          v = d_qStatesPtrs[idx];
                      else
                          v = DeviceComplex(0.);
                      d_qStatesPtrs[idx] = norm * v;
                  });
    }
    else {
        real norm = real(1.) / std::sqrt(real(1.) - prob);
        transform(begin, end,
                  [=]__device__(QstateIdx idx) mutable {
                      DeviceComplex v;
                      if ((idx & lane_bit) == 0)
                          v = DeviceComplex(0.);
                      else
                          v = d_qStatesPtrs[idx];
                      d_qStatesPtrs[idx] = norm * v;
                  });
    }
}


template<class real> void DeviceProcPrimitives<real>::
decohereAndShrink(DevicePtrs &dstDevPtrs,
                  int lane, int value, real prob, const DevicePtrs &srcDevPtrs,
                  QstateIdx begin, QstateIdx end) {
    QstateIdx lane_bit = Qone << lane;
    QstateIdx bitmask_hi = ~((lane_bit << 1) - 1);
    QstateIdx bitmask_lo = lane_bit - 1;

    device_.makeCurrent();
    if (value == 0) {
        real norm = real(1.) / std::sqrt(prob);
        transform(begin, end,
                  [=]__device__(QstateIdx idx) mutable {
                      QstateIdx idx_0 = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                      dstDevPtrs[idx] = norm * srcDevPtrs[idx_0];
                  });
    }
    else {
        real norm = real(1.) / std::sqrt(real(1.) - prob);
        transform(begin, end,
                  [=]__device__(QstateIdx idx) mutable {
                      QstateIdx idx_0 = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                      QstateIdx idx_1 = idx_0 | lane_bit;
                      dstDevPtrs[idx] = norm * srcDevPtrs[idx_1];
                  });
    }
}


template<class real>
void DeviceProcPrimitives<real>::applyReset(DevicePtrs &d_qStatesPtrs, int lane,
                                            qgate::QstateIdx begin, qgate::QstateIdx end) {
    QstateIdx lane_bit = Qone << lane;

    /* Assuming reset is able to be applyed after measurement.
     * Ref: https://quantumcomputing.stackexchange.com/questions/3908/possibility-of-a-reset-quantum-gate */
    device_.makeCurrent();
    transform(begin, end,
              [=]__device__(QstateIdx idx) mutable {
                  DeviceComplex v;
                  if ((idx & lane_bit) == 0) {
                      QstateIdx idx_hi = idx | lane_bit;
                      v = d_qStatesPtrs[idx_hi];
                  }
                  else {
                      v = DeviceComplex(0.);
                  }
                  d_qStatesPtrs[idx] = v;
              });
}

template<class real>
void DeviceProcPrimitives<real>::applyUnaryGate(const DeviceMatrix2x2C<real> &mat,
                                                DevicePtrs &d_qStatesPtrs, int lane,
                                                qgate::QstateIdx begin, qgate::QstateIdx end) {
    DeviceMatrix2x2C<real> dmat(mat);

    QstateIdx lane_bit = Qone << lane;
    QstateIdx bitmask_hi = ~((lane_bit << 1) - 1);
    QstateIdx bitmask_lo = lane_bit - 1;
    
    device_.makeCurrent();
    transform(begin, end,
              [=]__device__(QstateIdx idx) mutable {
                  typedef DeviceComplexType<real> DeviceComplex;
                  QstateIdx idx_0 = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                  QstateIdx idx_1 = idx_0 | lane_bit;
                  const DeviceComplex &qs0 = d_qStatesPtrs[idx_0];
                  const DeviceComplex &qs1 = d_qStatesPtrs[idx_1];
                  DeviceComplex qsout0 = dmat(0, 0) * qs0 + dmat(0, 1) * qs1;
                  DeviceComplex qsout1 = dmat(1, 0) * qs0 + dmat(1, 1) * qs1;
                  d_qStatesPtrs[idx_0] = qsout0;
                  d_qStatesPtrs[idx_1] = qsout1;
              });    
}

template<class real> void DeviceProcPrimitives<real>::
applyControlGate(const DeviceMatrix2x2C<real> &mat,
                 DevicePtrs &d_qStatesPtrs, const qgate::QstateIdxTable256 *d_bitPermTables,
                 qgate::QstateIdx controlBits, qgate::QstateIdx targetBit,
                 qgate::QstateIdx begin, qgate::QstateIdx end) {    
        
    DeviceMatrix2x2C<real> dmat(mat);
    transform(begin, end,
              [=]__device__(QstateIdx idx) mutable {
                  QstateIdx permuted = 0;
                  for (int iTable = 0; iTable < 6; ++iTable) {
                      int iByte = (idx >> (8 * iTable)) & 0xff;
                      permuted |= d_bitPermTables[iTable][iByte];
                  }
                  
                  QstateIdx idx_0 = permuted | controlBits;
                  QstateIdx idx_1 = idx_0 | targetBit;
                  
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
