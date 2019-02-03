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
    
    void allocate(CUDADeviceList &devices, int nLanes, int nLanesInDevice);
    
    void deallocate();
    
    int getNLanes() const {
        return nLanes_;
    }

    int getNLanesInChunk() const {
        return devPtr_.nLanesInChunk;
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
        return 1 << (nLanes_ - devPtr_.nLanesInChunk);
    }

private:
    int nLanes_;
    DevicePtr devPtr_;
    CUDADeviceList deviceList_;
    
    /* hidden copy ctor */
    CUDAQubitStates(const CUDAQubitStates &);
};

}
