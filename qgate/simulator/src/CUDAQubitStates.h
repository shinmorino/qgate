#pragma once

#include <map>
#include "Interfaces.h"
#include "DeviceTypes.h"
#include "DeviceQubitStates.h"
#include "DeviceSet.h"


namespace qgate_cuda {

class CUDADevices;


/* representing entangled qubits, or a single qubit or entangled qubits. */
template<class real>
class CUDAQubitStates : public qgate::QubitStates {
public:
    typedef DeviceComplexType<real> DeviceComplex;
    typedef DeviceQubitStates<real> DeviceQstates;
    
    CUDAQubitStates();

    ~CUDAQubitStates();
    
    void allocate(const qgate::IdList &qregIdList, DeviceSet &deviceSet, int nLanesInDevice);
    
    void deallocate(DeviceSet &deviceSet);
    
    int getNQregs() const {
        return (int)qregIdList_.size();
    }

    int getLane(int qregId) const;

    int getNLanesInDevice() const {
        return nLanesInDevice_;
    }
    
    DeviceComplex *getDevicePtr() {
        return devQstates_.d_qStatesPtrs[0];
    }

    const DeviceComplex *getDevicePtrs() const {
        return devQstates_.d_qStatesPtrs[0];
    }
    
    DeviceQubitStates<real> &getDeviceQubitStates() {
        return devQstates_;
    }
    
private:
    qgate::IdList qregIdList_;
    DeviceQstates devQstates_;
    int nDevices_;
    int nLanesInDevice_;
    
    /* hidden copy ctor */
    CUDAQubitStates(const CUDAQubitStates &);
};

}
