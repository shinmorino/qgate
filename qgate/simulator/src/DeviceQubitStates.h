#pragma once

#include "DeviceTypes.h"

namespace qgate_cuda {

template<class real>
struct DeviceQubitStates {
    typedef DeviceComplexType<real> DeviceComplex;
    DeviceComplex *d_qStatesPtrs[MAX_N_DEVICES];
    int laneToQregId[64];
    int nLanesInDevice;
    int nLanes;
};

}
