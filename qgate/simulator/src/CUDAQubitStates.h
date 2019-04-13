#pragma once

#include "Interfaces.h"
#include "DeviceTypes.h"
#include "CUDADevice.h"
#include "MultiChunkPtr.h"
#include "MultiDeviceMemoryStore.h"


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
    
    void allocate(int nLanes);
    
    void deallocate();

    const MultiDeviceChunk &getMultiChunk() const {
        return *mchunk_;
    }
    
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

    int getNumChunks() const {
        return mchunk_->getNChunks();
    }

private:
    int nLanes_;
    DevicePtr devPtr_;
    MultiDeviceChunk *mchunk_;
    
    /* hidden copy ctor */
    CUDAQubitStates(const CUDAQubitStates &);
};

}
