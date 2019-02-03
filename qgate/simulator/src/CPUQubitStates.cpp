#include "CPUQubitStates.h"
#include <string.h>
#include <algorithm>


using qgate::QstateIdx;
using namespace qgate_cpu;
using qgate::Qone;


template<class real>
CPUQubitStates<real>::CPUQubitStates() {
    if (sizeof(real) == sizeof(float))
        prec_ = qgate::precFP32;
    else
        prec_ = qgate::precFP64;
    qstates_ = NULL;
}

template<class real>
CPUQubitStates<real>::~CPUQubitStates() {
    deallocate();
}

template<class real>
void CPUQubitStates<real>::allocate(const int nLanes) {
    assert(qstates_ == NULL);
    nLanes_ = nLanes;
    nStates_ = Qone << nLanes;
    qstates_ = (Complex*)malloc(sizeof(Complex) * nStates_);
}
    
template<class real>
void CPUQubitStates<real>::deallocate() {
    if (qstates_ != NULL)
        free(qstates_);
    qstates_ = NULL;
}

template class CPUQubitStates<float>;
template class CPUQubitStates<double>;
