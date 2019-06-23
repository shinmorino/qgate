#pragma once

#include "DeviceQubitStates.h"
#include "CUDADevice.h"
#include <vector>

namespace qgate_cuda {

template<class real>
class DeviceProbArrayCalculator {
public:
    void setUp(const qgate::IdListList &laneTransformTables, const qgate::QubitStatesList &qsList,
               CUDADeviceList &devices);

    void tearDown();

    void run(real *array, int nLanes, int nHiddenLanes);

private:
    void runOneStepReduction(real *array, int nLanes, int nHiddenLanes);
    void runMultiStepReduction(real *array, int nLanes, int nHiddenLanes);

    typedef DeviceQubitStates<real> DeviceQstates;
    typedef std::vector<DeviceQstates*> DeviceQstatesPtrs;
    int nQstates_;
    DeviceQstatesPtrs d_qsPtrs_;
    CUDADeviceList *devices_;

    enum { nMaxLanesToReduce = 2 };
};

}
