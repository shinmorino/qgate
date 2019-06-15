#pragma once

#include "Interfaces.h"


namespace qgate_cpu {

using qgate::ComplexType;
using qgate::Matrix2x2C64;
using qgate::QubitStates;
using qgate::QstateIdx;
using qgate::QstateSize;
using qgate::MathOp;
using qgate::QubitStatesList;

template<class real>
class CPUQubitStates;


template<class real>
class CPUQubitProcessor : public qgate::QubitProcessor {
    typedef ComplexType<real> Complex;
    typedef qgate::MatrixType<Complex, 2> Matrix2x2CR;
public:
    CPUQubitProcessor();
    ~CPUQubitProcessor();

    virtual void reset();
    
    virtual void initializeQubitStates(qgate::QubitStates &qstates, int nLanes);
    
    virtual void resetQubitStates(qgate::QubitStates &qstates);

    virtual double calcProbability(const qgate::QubitStates &qstates, int localLane);

    virtual void join(qgate::QubitStates &qstates,
                      const QubitStatesList &qstatesList, int nNewLanes);
    
    virtual void decohere(int value, double prob, qgate::QubitStates &qstates, int localLane);
    
    virtual void decohereAndSeparate(int value, double prob,
                                     qgate::QubitStates &qstates0, qgate::QubitStates &qstates1,
                                     const qgate::QubitStates &qstates, int localLane);
    
    virtual void applyReset(QubitStates &qstates, int localLane);

    virtual void applyUnaryGate(const Matrix2x2C64 &mat, QubitStates &qstates, int localLane);

    virtual void applyControlGate(const Matrix2x2C64 &mat, QubitStates &qstates,
                                  const qgate::IdList &localControlLanes, int localTargetLane);

    virtual void getStates(void *array, QstateIdx arrayOffset,
                           MathOp op,
                           const qgate::IdList *laneTransTables, qgate::QstateIdx emptyLaneMask,
                           const QubitStatesList &qstatesList,
                           QstateSize nStates, QstateIdx begin, QstateIdx step);

    virtual void prepareProbArray(void *probs,
                                  const qgate::IdList *laneTransformTables,
                                  const QubitStatesList &qstatesList, int nLanes, int nHiddenLanes);

    virtual qgate::SamplingPool *createSamplingPool(const qgate::IdList *laneTransformTables,
                                                    const QubitStatesList &qstatesList,
                                                    int nLanes, int nHiddenLanes,
                                                    const qgate::IdList &emptyLanes);
    
private:
    template<class G>
    void run(int nLanes, int nInputBits,
             const qgate::IdList &bitShiftMap, const G &gatef);

    real _calcProbability(const CPUQubitStates<real> &qstates, int localLane);
    
    template<class R, class F>
    void qubitsGetValues(R *values, const F &func,
                         const qgate::IdList *laneTransTables, qgate::QstateIdx emptyLaneMask,
                         const QubitStatesList &qstatesList,
                         QstateSize nStates, QstateIdx begin, QstateIdx step);
};

}
