#pragma once

#include "DeviceTypes.h"
#include "BitPermTable.h"
#include "MultiChunkPtr.h"

namespace qgate_cuda {

template<class real>
struct DeviceQubitStates {
    
    typedef DeviceComplexType<real> DeviceComplex;
    typedef MultiChunkPtr<DeviceComplex> DevicePtr;
    
    enum { nTables = 6 };
    qgate::QstateIdxTable256 bitPermTable[nTables];
    DevicePtr ptr;
    
    void set(const DevicePtr &ptr, const qgate::BitPermTable &perm);
};

template<class real> inline void DeviceQubitStates<real>::
set(const DevicePtr &_ptr, const qgate::BitPermTable &perm) {
    ptr = _ptr;
    memcpy(bitPermTable, perm.getTables(), sizeof(qgate::QstateIdxTable256) * nTables);
}

}
