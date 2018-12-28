#pragma once

#include "Interfaces.h"
#include "DeviceTypes.h"
#include "CUDADevice.h"
#include "MultiChunkPtr.h"


namespace qgate_cuda {

class CUDADevices;


/* representing entangled qubits, or a single qubit or entangled qubits. */
template<class real>
class CUDAQubitStates : public qgate::QubitStates {
public:
    typedef DeviceComplexType<real> DeviceComplex;
    typedef MultiChunkPtr<DeviceComplex> DevicePtr;
    
    CUDAQubitStates();

    ~CUDAQubitStates();
    
    void allocate(const qgate::IdList &qregIdList, CUDADeviceList &devices, int nLanesInDevice);
    
    void deallocate();
    
    int getNQregs() const {
        return (int)qregIdList_.size();
    }

    int getLane(int qregId) const;

    int getNLanesInChunk() const {
        return nLanesInChunk_;
    }

    const qgate::IdList getQregIdList() const {
        return qregIdList_;
    }
    
    DevicePtr &getDevicePtr() {
        return devPtr_;
    }
    
    const DevicePtr &getDevicePtr() const {
        return devPtr_;
    }

    int getDeviceNumber(int idx) {
        return deviceList_[idx]->getDeviceNumber();
    }

    int getNumChunks() const {
        return 1 << (qregIdList_.size() - nLanesInChunk_);
    }

private:
    qgate::IdList qregIdList_;
    DevicePtr devPtr_;
    CUDADeviceList deviceList_;
    int nLanesInChunk_;
    
    /* hidden copy ctor */
    CUDAQubitStates(const CUDAQubitStates &);
};

}
