#pragma once

#include "Types.h"

namespace qgate {

struct QubitStates {

    virtual ~QubitStates() { }
    
    virtual void deallocate() = 0;

    virtual int getNLanes() const = 0;

    enum Precision getPrec() const { return prec_; }

protected:
    Precision prec_;
};


struct QubitProcessor {

    virtual ~QubitProcessor() { }

    virtual void reset() = 0;

    virtual void initializeQubitStates(qgate::QubitStates &qstates, int nLanes) = 0;
    
    virtual void resetQubitStates(qgate::QubitStates &qstates) = 0;

    virtual double calcProbability(const qgate::QubitStates &qstates, int localLane) = 0;

    virtual void join(qgate::QubitStates &qstates, const QubitStatesList &qstatesList, int nNewLanes) = 0;
    
    virtual void decohere(int value, double prob, qgate::QubitStates &qstates, int localLane) = 0;
    
    virtual void decohereAndSeparate(int value, double prob,
                                     qgate::QubitStates &qstates0, qgate::QubitStates &qstates1,
                                     const qgate::QubitStates &qstates, int localLane) = 0;
    
    virtual void applyReset(QubitStates &qstates, int localLane) = 0;

    virtual void applyUnaryGate(const Matrix2x2C64 &mat, QubitStates &qstates, int localLane) = 0;

    virtual void applyControlGate(const Matrix2x2C64 &mat, QubitStates &qstates,
                                  const IdList &localControlLanes, int localTargetLane) = 0;
};


struct QubitsStatesGetter {

    virtual ~QubitsStatesGetter() { }

    virtual void getStates(void *array, QstateIdx arrayOffset,
                           MathOp op,
                           const IdList *laneTransformTables, QstateIdx emptyLaneMask,
                           const QubitStatesList &qstatesList,
                           QstateSize nStates, QstateIdx begin, QstateIdx step) = 0;

    virtual void prepareProbArray(void *prob,
                                  const qgate::IdListList &laneTransformTables,
                                  const QubitStatesList &qstatesList,
                                  int nLanes, int nHiddenLanes) = 0;

    virtual struct SamplingPool *createSamplingPool(const IdListList &laneTransformTables,
                                                    const QubitStatesList &qstatesList,
                                                    int nLanes, int nHiddenLanes,
                                                    const IdList &emptyLanes) = 0;
};


struct SamplingPool {

    virtual ~SamplingPool() { }

    virtual void sample(QstateIdx *observations, int nSamples, const double *randNum) = 0;

};

}
