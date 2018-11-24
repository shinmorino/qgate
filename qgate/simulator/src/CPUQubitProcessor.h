#pragma once

#include "Interfaces.h"


namespace qgate_cpu {

using qgate::ComplexType;
using qgate::Matrix2x2C64;
using qgate::QubitStates;
using qgate::QstateIdx;
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

    virtual void prepare(qgate::QubitStates &qstates);
    
    virtual int measure(double randNum, QubitStates &qstates, int qregId) const;
    
    virtual void applyReset(QubitStates &qstates, int qregId) const;

    virtual void applyUnaryGate(const Matrix2x2C64 &mat, QubitStates &qstates, int qregId) const;

    virtual void applyControlGate(const Matrix2x2C64 &mat, QubitStates &qstates, int controlId, int targetId) const;

   virtual void getStates(void *array, QstateIdx arrayOffset,
                          MathOp op,
                          const QubitStatesList &qstatesList,
                          QstateIdx beginIdx, QstateIdx endIdx) const;

private:
    template<class R, class F>
    void qubitsGetValues(R *values, const F &func,
                         const QubitStatesList &qstatesList,
                         QstateIdx beginIdx, QstateIdx endIdx) const;

};

}
