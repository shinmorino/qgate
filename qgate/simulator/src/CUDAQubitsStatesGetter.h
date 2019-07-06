#pragma once
#include "Types.h"
#include "Interfaces.h"

namespace qgate_cuda {

template<class real>
class CUDAQubitsStatesGetter : public qgate::QubitsStatesGetter {

    void getStates(void *array, qgate::QstateIdx arrayOffset,
                   qgate::MathOp op,
                   const qgate::IdList *laneTransTables, qgate::QstateIdx emptyLaneMask,
                   const qgate::QubitStatesList &qstatesList,
                   qgate::QstateIdx nStates, qgate::QstateIdx begin, qgate::QstateIdx step);
    
    void prepareProbArray(void *_prob,
                          const qgate::IdListList &laneTransformTables,
                          const qgate::QubitStatesList &qstatesList,
                          int nLanes, int nHiddenLanes);
    
    qgate::SamplingPool *
    createSamplingPool(const qgate::IdListList &laneTransformTables,
                       const qgate::QubitStatesList &qstatesList,
                       int nLanes, int nHiddenLanes, const qgate::IdList &emptyLanes);
};
    
}
