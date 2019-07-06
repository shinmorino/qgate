#include "CUDAQubitsStatesGetter.h"
#include "Types.h"
#include "Parallel.h"
#include "BitPermTable.h"
#include "CUDAQubitStates.h"
#include "DeviceGetStates.h"
#include "DeviceProbArrayCalculator.h"
#include "CUDAGlobals.h"
#include "CPUSamplingPool.h"

using qgate::QstateIdx;
using qgate::QstateSize;
using qgate::MathOp;
using qgate::QubitStatesList;
using qgate_cuda::CUDAQubitStates;

using qgate::Qone;

namespace qcuda = qgate_cuda;
using namespace qgate_cuda;

template<class real> void CUDAQubitsStatesGetter<real>::
getStates(void *array, QstateIdx arrayOffset,
          MathOp op,
          const qgate::IdList *laneTransTables, QstateIdx emptyLaneMask,
          const QubitStatesList &qstatesList,
          QstateIdx nStates, QstateIdx begin, QstateIdx step) {

    for (int idx = 0; idx < (int)qstatesList.size(); ++idx) {
        const qgate::QubitStates *qstates = qstatesList[idx];
        if (sizeof(real) == sizeof(float)) {
            abortIf(qstates->getPrec() != qgate::precFP32, "Wrong type");
        }
        else if (sizeof(real) == sizeof(double)) {
            abortIf(qstates->getPrec() != qgate::precFP64, "Wrong type");
        }
    }

    CUDADeviceList &devices = cudaDevices.devices();
    DeviceGetStates<real> getStates(laneTransTables, emptyLaneMask, qstatesList, devices);
    getStates.run(array, arrayOffset, op, nStates, begin, step);
}

template<class real> void CUDAQubitsStatesGetter<real>::
prepareProbArray(void *prob,
                 const qgate::IdListList &laneTransformTables,
                 const QubitStatesList &qstatesList,
                 int nLanes, int nHiddenLanes) {
    for (int idx = 0; idx < (int)qstatesList.size(); ++idx) {
        const qgate::QubitStates *qstates = qstatesList[idx];
        if (sizeof(real) == sizeof(float)) {
            abortIf(qstates->getPrec() != qgate::precFP32, "Wrong type");
        }
        else if (sizeof(real) == sizeof(double)) {
            abortIf(qstates->getPrec() != qgate::precFP64, "Wrong type");
        }
    }
    CUDADeviceList &devices = cudaDevices.devices();
    if (nHiddenLanes == 0) {
        DeviceGetStates<real> getStates(laneTransformTables.data(),
                                        0, qstatesList, devices);
        QstateSize nStates = Qone << nLanes;
        getStates.run(prob, 0, qgate::mathOpProb, nStates, 0, 1);
    }
    else {
        /* FIXME: activeDevices_ ? */
        DeviceProbArrayCalculator<real> getProbArray;
        getProbArray.setUp(laneTransformTables, qstatesList, devices);
        getProbArray.run(static_cast<real*>(prob), nLanes, nHiddenLanes);
        getProbArray.tearDown();
    }
}

template<class real> qgate::SamplingPool *CUDAQubitsStatesGetter<real>::
createSamplingPool(const qgate::IdListList &laneTransformTables,
                   const QubitStatesList &qstatesList,
                   int nLanes, int nHiddenLanes, const qgate::IdList &emptyLanes) {
    /* FIXME: implement CUDA sampling pool. */
    QstateSize nStates = Qone << nLanes;
    real *prob = (real*)malloc(sizeof(real) * nStates);
    prepareProbArray(prob, laneTransformTables, qstatesList, nLanes, nHiddenLanes);
    return new qgate_cpu::CPUSamplingPool<real>(prob, nLanes, emptyLanes);
}

template class CUDAQubitsStatesGetter<float>;
template class CUDAQubitsStatesGetter<double>;
