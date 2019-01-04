#pragma once

#include "DeviceSum.h"
#include "MultiChunkPtr.h"

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
    
    void measure_set0(DevicePtrs &devPtrs, int lane, real prob,
                      qgate::QstateIdx begin, qgate::QstateIdx end);
    
    void measure_set1(DevicePtrs &devPtrs, int lane, real prob,
                      qgate::QstateIdx begin, qgate::QstateIdx end);
    
    void applyReset(DevicePtrs &devPtrs, int lane,
                    qgate::QstateIdx begin, qgate::QstateIdx end);
    
    void applyUnaryGate(const DeviceMatrix2x2C<real> &mat,
                        DevicePtrs &devPtrs, int lane,
                        qgate::QstateIdx begin, qgate::QstateIdx end);
    
    void applyControlGate(const DeviceMatrix2x2C<real> &mat,
                          DevicePtrs &devPtrs, int controlLane, int targetLane,
                          qgate::QstateIdx begin, qgate::QstateIdx end);
    
private:
    DeviceSum<real> deviceSum_;
    CUDADevice &device_;

    DeviceProcPrimitives(const DeviceProcPrimitives &);
};

}
