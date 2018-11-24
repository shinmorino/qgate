#pragma once

#include "Types.h"

namespace qgate {

struct QubitStates {

    virtual ~QubitStates() { }
    
    virtual void allocate(const IdList &qregIdList) = 0;
    
    virtual void deallocate() = 0;

    virtual void reset() = 0;
    
    virtual int getNQregs() const = 0;

    enum Precision getPrec() const { return prec_; }

protected:
    Precision prec_;
};


struct QubitProcessor {

    virtual ~QubitProcessor() { }

    virtual void prepare(qgate::QubitStates &qstates) = 0;
    
    virtual int measure(double randNum, qgate::QubitStates &qstates, int qregId) const = 0;
    
    virtual void applyReset(QubitStates &qstates, int qregId) const = 0;

    virtual void applyUnaryGate(const Matrix2x2C64 &mat, QubitStates &qstates, int qregId) const = 0;

    virtual void applyControlGate(const Matrix2x2C64 &mat, QubitStates &qstates, int controlId, int targetId) const = 0;

   virtual void getStates(void *array, QstateIdx arrayOffset,
                          MathOp op,
                          const QubitStatesList &qstatesList,
                          QstateIdx beginIdx, QstateIdx endIdx) const = 0;

};

}
