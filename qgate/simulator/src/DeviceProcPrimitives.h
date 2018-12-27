#pragma once

#include "DeviceQubitStates.h"
#include "DeviceSum.cuh"

namespace qgate_cuda {

template<class real>
class DeviceProcPrimitives {
public:
    typedef DeviceQubitStates<real> DevQubitStates;
    typedef DeviceComplexType<real> DeviceComplex;
    typedef qgate::ComplexType<real> Complex;
    
    DeviceProcPrimitives(CUDADevice &device);

    void synchronize();

    void set(DevQubitStates &qStates,
             const void *pv, qgate::QstateIdx offset, qgate::QstateSize size);

    void fillZero(DevQubitStates &qStates, qgate::QstateIdx begin, qgate::QstateIdx end);
    
    void traceOut_launch(DevQubitStates &qStates, int lane,
                         qgate::QstateIdx begin, qgate::QstateIdx end);
    
    real traceOut_sync();
    
    void measure_set0(DevQubitStates &qStates, int lane, real prob,
                      qgate::QstateIdx begin, qgate::QstateIdx end);
    
    void measure_set1(DevQubitStates &qStates, int lane, real prob,
                      qgate::QstateIdx begin, qgate::QstateIdx end);
    
    void applyReset(DevQubitStates &qStates, int lane,
                    qgate::QstateIdx begin, qgate::QstateIdx end);
    
    void applyUnaryGate(const DeviceMatrix2x2C<real> &mat,
                        DevQubitStates &qStates, int lane,
                        qgate::QstateIdx begin, qgate::QstateIdx end);
    
    void applyControlGate(const DeviceMatrix2x2C<real> &mat,
                          DevQubitStates &qStates, int controlLane, int targetLane,
                          qgate::QstateIdx begin, qgate::QstateIdx end);

    void getStates(void *array, qgate::QstateIdx arrayOffset,
                   qgate::MathOp op,
                   const DevQubitStates &devQstates,
                   qgate::QstateIdx beginIdx, qgate::QstateIdx endIdx);
    
    template<class R, class F>
    void getStates(R *values, const F &op,
                   const DevQubitStates &qstates,
                   qgate::QstateIdx begin, qgate::QstateIdx end);
    
private:
    DeviceSum<real> deviceSum_;
    CUDADevice &device_;

    DeviceProcPrimitives(const DeviceProcPrimitives &);
};

}
