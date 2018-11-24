#pragma once

#include "Interfaces.h"

namespace qgate_cpu {

using qgate::QstateIdx;
using qgate::ComplexType;
using qgate::IdList;
using qgate::Matrix2x2C64;

    
/* representing entangled qubits, or a single qubit or entangled qubits. */
template<class real>
class CPUQubitStates : public qgate::QubitStates {
    typedef ComplexType<real> Complex;
public:
    CPUQubitStates();

    ~CPUQubitStates();
    
    void allocate(const IdList &qregIdList);
    
    void deallocate();

    void reset();

    int getNQregs() const {
        return (int)qregIdList_.size();
    }

    int getLane(int qregId) const;

    /* CPUQubitStates-specific methods */
    
    Complex &operator[](QstateIdx idx) {
        return qstates_[idx];
    }
    
    const Complex &operator[](QstateIdx idx) const {
        return qstates_[idx];
    }

    const Complex &getStateByGlobalIdx(QstateIdx idx) const;

    QstateIdx convertToLocalLaneIdx(QstateIdx idx) const;
    
private:
    QstateIdx nStates_;
    IdList qregIdList_;
    Complex *qstates_;
    
    /* hidden copy ctor */
    CPUQubitStates(const CPUQubitStates &);
};

}
