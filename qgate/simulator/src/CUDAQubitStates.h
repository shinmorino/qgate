#pragma once

#include <map>
#include "DeviceTypes.h"
#include "Interfaces.h"

namespace qgate_cuda {

class CUDADevice;

template<class real>
struct DeviceQubitStates {
    typedef DeviceComplexType<real> DeviceComplex;
    
    __host__
    DeviceQubitStates() : d_qregIdList_(NULL), nQregIds_(-1), d_qstates_(NULL) { }

    __host__
    void allocate(const qgate::IdList &qregIdList, CUDADevice &device);

    __host__
    void deallocate(CUDADevice &device);

    __host__
    void reset();

    __host__
    qgate::QstateIdx getNStates() const;
    
    int *d_qregIdList_;
    int nQregIds_;
    DeviceComplex *d_qstates_;
    qgate::QstateIdx nStates_;
};


/* representing entangled qubits, or a single qubit or entangled qubits. */
template<class real>
class CUDAQubitStates : public qgate::QubitStates {
public:
    typedef DeviceComplexType<real> DeviceComplex;
    
    CUDAQubitStates();

    ~CUDAQubitStates();

    void setDevice(CUDADevice *device);
    
    void allocate(const qgate::IdList &qregIdList);
    
    void deallocate();

    void reset();
    
    int getNQregs() const {
        return (int)qregIdList_.size();
    }

    int getLane(int qregId) const;

    DeviceComplex *getDevicePtr() {
        return devQstates_.d_qstates_;
    }

    const DeviceComplex *getDevicePtr() const {
        return devQstates_.d_qstates_;
    }

    const DeviceQubitStates<real> &getDeviceQubitStates() const {
        return devQstates_;
    }
    
private:
    CUDADevice *device_;
    qgate::IdList qregIdList_;
    DeviceQubitStates<real> devQstates_;
    
    /* hidden copy ctor */
    CUDAQubitStates(const CUDAQubitStates &);
};

}
