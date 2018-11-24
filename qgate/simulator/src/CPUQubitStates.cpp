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
void CPUQubitStates<real>::allocate(const IdList &qregIdList) {
    qregIdList_ = qregIdList;
    assert(qstates_ == NULL);
    nStates_ = Qone << qregIdList_.size();
    qstates_ = (Complex*)malloc(sizeof(Complex) * nStates_);
    reset();
}
    
template<class real>
void CPUQubitStates<real>::deallocate() {
    if (qstates_ != NULL)
        free(qstates_);
    qstates_ = NULL;
}

template<class real>
void CPUQubitStates<real>::reset() {
    memset(qstates_, 0, sizeof(Complex) * nStates_);
    qstates_[0] = Complex(1.);
}

template<class real>
int CPUQubitStates<real>::getLane(int qregId) const {
    IdList::const_iterator it = std::find(qregIdList_.begin(), qregIdList_.end(), qregId);
    assert(it != qregIdList_.end());
    return (int)std::distance(qregIdList_.begin(), it);
}

template<class real>
const ComplexType<real> &CPUQubitStates<real>::getStateByGlobalIdx(QstateIdx idx) const {
    QstateIdx localIdx = convertToLocalLaneIdx(idx);
    return qstates_[localIdx];
}

template<class real>
QstateIdx CPUQubitStates<real>::convertToLocalLaneIdx(QstateIdx globalIdx) const {
    QstateIdx localIdx = 0;
    for (int bitPos = 0; bitPos < (int)qregIdList_.size(); ++bitPos) {
        int qregId = qregIdList_[bitPos]; 
        if ((Qone << qregId) & globalIdx)
            localIdx |= Qone << bitPos;
    }
    return localIdx;
}


template class CPUQubitStates<float>;
template class CPUQubitStates<double>;
