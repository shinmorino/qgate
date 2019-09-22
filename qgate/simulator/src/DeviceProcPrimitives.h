#pragma once

#include "DeviceSum.h"
#include "MultiChunkPtr.h"
#include "BitPermTable.h"

namespace qgate_cuda {

template<class real>
class DeviceProcPrimitives {
public:
    typedef DeviceComplexType<real> DeviceComplex;
    typedef MultiChunkPtr<DeviceComplex> DevicePtrs;
    typedef qgate::ComplexType<real> Complex;
    
    DeviceProcPrimitives(CUDADevice &device);

    CUDADevice &device() {
        return device_;
    }

    void set(DevicePtrs &devPtrs,
             const void *pv, qgate::QstateIdx offset, qgate::QstateSize size);

    void fillZero(DevicePtrs &devPtrs, qgate::QstateIdx begin, qgate::QstateIdx end);
    
    void calcProb_launch(const DevicePtrs &devPtrs, int lane,
                         qgate::QstateIdx begin, qgate::QstateIdx end);
    
    real calcProb_sync();

    void copyAndFillZero(DevicePtrs &dstPtrs,
                         const DevicePtrs &srcPtrs, qgate::QstateIdx srcSize,
                         qgate::QstateIdx begin, qgate::QstateIdx end);

    void kron(DevicePtrs &dstPtrs,
              const DevicePtrs &srcPtrs0, qgate::QstateSize Nsrc0,
              const DevicePtrs &srcPtrs1, qgate::QstateSize Nsrc1,
              qgate::QstateIdx begin, qgate::QstateIdx end);

    void kronInPlace_0(DevicePtrs &dstPtrs, qgate::QstateSize Ndst,
                       const DevicePtrs &srcPtrs, qgate::QstateSize Nsrc,
                       qgate::QstateIdx begin, qgate::QstateIdx end);

    void kronInPlace_1(DevicePtrs &dstPtrs, qgate::QstateSize Ndst,
                       const DevicePtrs &srcPtrs, qgate::QstateSize Nsrc,
                       qgate::QstateIdx begin, qgate::QstateIdx end);
    
    void decohere(DevicePtrs &devPtrs, int lane, int value, real prob,
                  qgate::QstateIdx begin, qgate::QstateIdx end);
    
    void decohereAndShrink(DevicePtrs &dstDevPtrs,
                           int lane, int value, real prob, const DevicePtrs &srcDevPtrs,
                           qgate::QstateIdx begin, qgate::QstateIdx end);
    
    void applyReset(DevicePtrs &devPtrs, int lane,
                    qgate::QstateIdx begin, qgate::QstateIdx end);
    
    void applyGate(const DeviceMatrix2x2C<real> &mat,
                   DevicePtrs &devPtrs, int lane,
                   qgate::QstateIdx begin, qgate::QstateIdx end);
    
    void applyControlledGate(const DeviceMatrix2x2C<real> &mat,
                             DevicePtrs &devPtrs, const qgate::QstateIdxTable256 *d_bitPermTables,
                             qgate::QstateIdx controlBits, qgate::QstateIdx targetBit,
                             qgate::QstateIdx begin, qgate::QstateIdx end);
    
private:
    DeviceSum<real> deviceSum_;
    CUDADevice &device_;

    DeviceProcPrimitives(const DeviceProcPrimitives &);
};

}
