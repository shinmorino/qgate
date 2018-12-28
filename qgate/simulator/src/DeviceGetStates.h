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
    
    DeviceGetStates(const qgate::QubitStatesList &qstatesList, CUDADeviceList &activeDevices);
    ~DeviceGetStates();
    
    void run(void *array, qgate::QstateIdx arrayOffset, qgate::MathOp op,
             qgate::QstateIdx begin, qgate::QstateIdx end);
    
    template<class R, class F>
    void run(R *values, const F &op,
             qgate::QstateIdx begin, qgate::QstateIdx end);

    CUDADeviceList activeDevices_;
    qgate::QstateIdx begin_, end_;
    qgate::QstateIdx pos_;
    int stride_;
    
    /* Context */
    struct IdList {
        int id[63];
        int size;
    };

    struct GetStatesContext {
        qgate::QstateIdx begin, end;
        IdList *d_idLists;
        int nQstates;
        DevicePtr *d_qStatesPtr;
        void *h_values;
        CUDADevice *device;
    };

    GetStatesContext *contexts_;
    
    template<class R, class F>
    bool launch(GetStatesContext &ctx, const F &op);
    template<class R>
    void syncAndCopy(R *values, GetStatesContext &ctx);
};

}

