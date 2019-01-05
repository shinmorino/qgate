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
    perm_.init_QregIdxToLocalIdx(qregIdList_);
}
    
template<class real>
void CPUQubitStates<real>::deallocate() {
    if (qstates_ != NULL)
        free(qstates_);
    qstates_ = NULL;
}

template<class real>
int CPUQubitStates<real>::getLane(int qregId) const {
    IdList::const_iterator it = std::find(qregIdList_.begin(), qregIdList_.end(), qregId);
    assert(it != qregIdList_.end());
    return (int)std::distance(qregIdList_.begin(), it);
}

template<class real>
const ComplexType<real> &CPUQubitStates<real>::getStateByQregIdx(QstateIdx idx) const {
    QstateIdx localIdx = convertToLocalLaneIdx(idx);
    return qstates_[localIdx];
}

template<class real>
QstateIdx CPUQubitStates<real>::convertToLocalLaneIdx(QstateIdx qregIdx) const {
    return perm_.permute(qregIdx);
}

template<class real>
void CPUQubitStates<real>::getStateByQregIdx256(ComplexType<real> *qStates, QstateIdx qregIdx) const {
    throwErrorIf((qregIdx % 256) != 0, "globalIdx should be a multiple of 256.");
    QstateIdx laneIdx56 = perm_.permute_56bits(qregIdx);

    for (int idx = 0; idx < 256; ++idx) {
        QstateIdx localIdx = perm_.permute_8bits(laneIdx56, idx);
        qStates[idx] = qstates_[localIdx];
    }
}

template class CPUQubitStates<float>;
template class CPUQubitStates<double>;
