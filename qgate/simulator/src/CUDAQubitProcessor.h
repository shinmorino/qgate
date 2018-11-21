#pragma once

#include "Interfaces.h"
#include "DeviceTypes.h"

namespace qgate_cuda {

using qgate::Matrix2x2C64;
using qgate::QubitStates;
using qgate::MathOp;
using qgate::QstateIdxType;
using qgate::QubitStatesList;

class CUDAResource;

template<class real>
class CUDAQubitProcessor : public qgate::QubitProcessor {
    typedef DeviceComplexType<real> DeviceComplex;
public:
    CUDAQubitProcessor(CUDAResource &rsrc);
    ~CUDAQubitProcessor();

    virtual void prepare(qgate::QubitStates &qstates);
    
    virtual int measure(double randNum, QubitStates &qstates, int qregId) const;
    
    virtual void applyReset(QubitStates &qstates, int qregId) const;

    virtual void applyUnaryGate(const Matrix2x2C64 &mat, QubitStates &qstates, int qregId) const;

    virtual void applyControlGate(const Matrix2x2C64 &mat, QubitStates &qstates, int controlId, int targetId) const;

    virtual void getQubitStates(void *values, QstateIdxType beginIdx, QstateIdxType endIdx, MathOp op) const;
    
    virtual void getStates(void *array, QstateIdxType arrayOffset,
                           MathOp op,
                           const QubitStatesList &qstatesList,
                           QstateIdxType beginIdx, QstateIdxType endIdx) const;
    
    
    template<class F>
    void getValues(real *values, QstateIdxType arrayOffset,
                   MathOp op,
                   const QubitStatesList &qstatesList,
                   QstateIdxType beginIdx, QstateIdxType endIdx,
                   const F &func) const;

private:
    CUDAResource &rsrc_;
};
        
}
