#pragma once

#include "Interfaces.h"
#include "DeviceTypes.h"
#include "CUDADevice.h"
#include "MultiChunkPtr.h"
#include <map>

namespace qgate_cuda {

using qgate::Matrix2x2C64;
using qgate::QubitStates;
using qgate::MathOp;
using qgate::QstateIdx;
using qgate::QstateSize;
using qgate::QubitStatesList;

/* forwarded decls. */
template<class R> class CUDAQubitStates;
template<class R> class DeviceProcPrimitives;


template<class real>
class CUDAQubitProcessor : public qgate::QubitProcessor {
    typedef qgate::ComplexType<real> Complex;
    typedef DeviceComplexType<real> DeviceComplex;
    typedef CUDAQubitStates<real> CUQStates;
    typedef MultiChunkPtr<DeviceComplex> DevicePtr;
public:
    CUDAQubitProcessor(CUDADevices &devices);
    ~CUDAQubitProcessor();

    virtual void reset();

    virtual void initializeQubitStates(qgate::QubitStates &qstates, int nLanes);

    virtual void resetQubitStates(qgate::QubitStates &qstates);

    virtual double calcProbability(const qgate::QubitStates &qstates, int qregId);

    virtual void join(qgate::QubitStates &qstates,
                      const QubitStatesList &qstatesList, int nNewLanes);
    
    virtual void decohere(int value, double prob, qgate::QubitStates &qstates, int localLane);
    
    virtual void decohereAndSeparate(int value, double prob,
                                     qgate::QubitStates &qstates0, qgate::QubitStates &qstates1,
                                     const qgate::QubitStates &qstates, int localLane);
    
    virtual void applyReset(QubitStates &qstates, int qregId);

    virtual void applyUnaryGate(const Matrix2x2C64 &mat, QubitStates &qstates, int qregId);

    virtual void applyControlGate(const Matrix2x2C64 &mat, QubitStates &qstates,
                                  const qgate::IdList &localControlLanes, int targetId);

    virtual void getStates(void *array, QstateIdx arrayOffset,
                           MathOp op,
                           const qgate::IdList *laneTransTables, qgate::QstateIdx emptyLaneMask,
                           const QubitStatesList &qstatesList,
                           QstateIdx beginIdx, QstateIdx endIdx, QstateIdx step);

    virtual void prepareProbArray(void *prob,
                                  const qgate::IdListList &laneTransformTables,
                                  const qgate::QubitStatesList &qstatesList,
                                  int nLanes, int nHiddenLanes);

    virtual qgate::SamplingPool *createSamplingPool(const qgate::IdListList &laneTransformTables,
                                                    const qgate::QubitStatesList &qstatesList,
                                                    int nLanes, int nHiddenLanes,
                                                    const qgate::IdList &emptyLanes);
	
    /* synchronize all active devices */
    void synchronize();

    /* synchronize when using multiple devices */
    void synchronizeMultiDevice();

    /* template methods to use device lambda.
     * They're intented for private use, though placed on public space to enable device lambda. */
    template<class F>
    void distribute(int nProcs, const F &f, QstateSize nThreads);

    template<class F>
    void distribute(const CUQStates &cuQstates, const F &f);

    template<class F>
    void distribute(const CUQStates &cuQstates, const F &f, QstateIdx begin, QstateIdx end);

private:
    real _calcProbability(const CUDAQubitStates<real> &qstates, int lane);

    CUDADevices &devices_;
    typedef std::vector<DeviceProcPrimitives<real>*> Procs;
    Procs procs_;
    CUDADeviceList activeDevices_;
};
        
}
