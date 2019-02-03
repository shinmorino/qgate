#pragma once

#include "Interfaces.h"

namespace qgate_cpu {

using qgate::QstateIdx;
using qgate::QstateSize;
using qgate::ComplexType;
using qgate::Matrix2x2C64;

    
/* representing entangled qubits, or a single qubit or entangled qubits. */
template<class real>
class CPUQubitStates : public qgate::QubitStates {
    typedef ComplexType<real> Complex;
public:
    CPUQubitStates();

    ~CPUQubitStates();
    
    void allocate(const int nLanes);
    
    void deallocate();

    int getNLanes() const {
        return nLanes_;
    }

    /* CPUQubitStates-specific methods */

    Complex *getPtr() { return qstates_; }

    const Complex *getPtr() const { return qstates_; }
    
    Complex &operator[](QstateIdx idx) {
        assert(0 <= idx);
        assert(idx < nStates_);
        return qstates_[idx];
    }
    
    const Complex &operator[](QstateIdx idx) const {
        assert(0 <= idx);
        assert(idx < nStates_);
        return qstates_[idx];
    }

private:
    QstateSize nStates_;
    int nLanes_;
    Complex *qstates_;
    
    /* hidden copy ctor */
    CPUQubitStates(const CPUQubitStates &);
};

}
