#pragma once

#include "CUDAQubitStates.h"
#include "CUDADevice.h"
#include "DeviceTypes.h"
#include "MultiChunkPtr.h"
#include "Interfaces.h"


namespace qgate_cuda {

template<class real>
struct DeviceGetStates {
    typedef DeviceComplexType<real> DeviceComplex;
    typedef MultiChunkPtr<DeviceComplex> DevicePtr;
    typedef qgate::ComplexType<real> Complex;
    
    DeviceGetStates(const qgate::IdList *extToLocal, qgate::QstateIdx emptyLaneMask,
                    const qgate::QubitStatesList &qstatesList,
                    CUDADeviceList &activeDevices);
    ~DeviceGetStates();
    
    void run(void *array, qgate::QstateIdx arrayOffset, qgate::MathOp op,
             qgate::QstateSize nStates, qgate::QstateIdx begin, qgate::QstateIdx step);
    
    template<class R, class F>
    void run(R *values, const F &op,
             qgate::QstateSize nStates, qgate::QstateIdx begin, qgate::QstateIdx step);

    qgate::QstateIdx emptyLaneMask_;
    CUDADeviceList activeDevices_;
    qgate::QstateIdx start_, step_;
    qgate::QstateIdx pos_;
    qgate::QstateSize nStates_;
    int stride_;
    
    /* Context */
    struct LaneTransform {
        int externalLanes[63];
        int size;
    };

    struct DeviceGetStatesContext {
        qgate::QstateIdx begin, end;
        LaneTransform *d_laneTrans;
        int nQstates;
        DevicePtr *d_qStatesPtr;
        void *h_values;
    };

    struct GetStatesContext {
        DeviceGetStatesContext dev;
        CUDADevice *device;
        cudaEvent_t event;
    };

    typedef std::vector<GetStatesContext> Contexts;
    Contexts contexts_;
    enum { nContextsPerDevice = 2 };

    template<class R, class F>
    bool launch(GetStatesContext &ctx, const F &op);
    template<class R>
    void syncAndCopy(R *values, GetStatesContext &ctx);
};

}

