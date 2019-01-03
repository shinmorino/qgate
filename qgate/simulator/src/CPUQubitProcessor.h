#pragma once

#include "Interfaces.h"
#include "Parallel.h"


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

    virtual void clear();
    
    virtual void prepare();
    
    virtual void initializeQubitStates(const qgate::IdList &qregIdList, qgate::QubitStates &qstates,
                                       int nLanesPerDevice, qgate::IdList &_deviceIds);
    
    virtual void resetQubitStates(qgate::QubitStates &qstates);
    
    virtual int measure(double randNum, QubitStates &qstates, int qregId);
    
    virtual void applyReset(QubitStates &qstates, int qregId);

    virtual void applyUnaryGate(const Matrix2x2C64 &mat, QubitStates &qstates, int qregId);

    virtual void applyControlGate(const Matrix2x2C64 &mat, QubitStates &qstates, int controlId, int targetId);

   virtual void getStates(void *array, QstateIdx arrayOffset,
                          MathOp op,
                          const QubitStatesList &qstatesList,
                          QstateIdx beginIdx, QstateIdx endIdx);

private:
    template<class R, class F>
    void qubitsGetValues(R *values, const F &func,
                         const QubitStatesList &qstatesList,
                         QstateIdx beginIdx, QstateIdx endIdx);

    qgate::Parallel parallel_;
};

}
