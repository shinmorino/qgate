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

    virtual void initializeQubitStates(qgate::QubitStates &qstates,
                                       int nLanes, int nLanesPerDevice, qgate::IdList &_deviceIds) = 0;
    
    virtual void resetQubitStates(qgate::QubitStates &qstates) = 0;

    virtual double calcProbability(const qgate::QubitStates &qstates, int localLane) = 0;
    
    virtual int measure(double randNum, qgate::QubitStates &qstates, int localLane) = 0;
    
    virtual void applyReset(QubitStates &qstates, int localLane) = 0;

    virtual void applyUnaryGate(const Matrix2x2C64 &mat, QubitStates &qstates, int localLane) = 0;

    virtual void applyControlGate(const Matrix2x2C64 &mat, QubitStates &qstates,
                                  const IdList &localControlLanes, int localTargetLane) = 0;

    virtual void getStates(void *array, QstateIdx arrayOffset,
                           MathOp op,
                           const IdList *laneTransformTables, const QubitStatesList &qstatesList,
                           QstateSize nStates, QstateIdx begin, QstateIdx step) = 0;

};

}
