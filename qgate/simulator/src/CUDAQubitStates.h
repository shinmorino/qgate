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

    void deallocate();

    void setMultiDeviceChunk(MultiDeviceChunk *mchunk, int nLanes);

    int getNLanes() const {
        return nLanes_;
    }

    int getNLanesPerChunk() const {
        return mchunk_->getPo2Idx() - (2 + sizeof(DeviceComplex) / 8);
    }

    DevicePtr &getDevicePtr() {
        return devPtr_;
    }
    
    const DevicePtr &getDevicePtr() const {
        return devPtr_;
    }

private:
    int nLanes_;
    DevicePtr devPtr_;
    MultiDeviceChunk *mchunk_;

    /* hidden copy ctor */
    CUDAQubitStates(const CUDAQubitStates &);
};

}
