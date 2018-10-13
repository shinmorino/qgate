#include "CPURuntime.h"
#include "loop.h"
#include <string.h>
#include <algorithm>


namespace {

const QstateIdxType One = 1;
const QstateIdxType Two = 2;

}


CPUQubits::~CPUQubits() {
}


void CPUQubits::addQubitStates(int key, CPUQubitStates *qstates) {
    cpuQubitStatesMap_[key] = qstates;
}

void CPUQubits::detachQubitStates() {
    cpuQubitStatesMap_.clear();
}
    
CPUQubitStates &CPUQubits::operator[](int key) {
    CPUQubitStatesMap::iterator it = cpuQubitStatesMap_.find(key);
    assert(it != cpuQubitStatesMap_.end());
    return *it->second;
}

const CPUQubitStates &CPUQubits::operator[](int key) const {
    CPUQubitStatesMap::const_iterator it = cpuQubitStatesMap_.find(key);
    assert(it != cpuQubitStatesMap_.end());
    return *it->second;
}

QstateIdxType CPUQubits::getNStates() const {
    return One << cpuQubitStatesMap_.size();
}

namespace {


template<class R>
inline R abs2(const std::complex<R> &c) {
    return c.real() * c.real() + c.imag() * c.imag();
}

template<class R>
inline std::complex<R> null(const std::complex<R> &c) {
    return c;
}


}


template<class V, class F>
void CPUQubits::getValues(V *values,
                          QstateIdxType beginIdx, QstateIdxType endIdx,
                          const F &func) const {
    
    size_t nQubitStates = cpuQubitStatesMap_.size();
    CPUQubitStates **qstates = new CPUQubitStates*[nQubitStates];
    CPUQubitStatesMap::const_iterator it = cpuQubitStatesMap_.begin();
    for (int qstatesIdx = 0; (int)qstatesIdx < (int)cpuQubitStatesMap_.size(); ++qstatesIdx) {
        qstates[qstatesIdx] = it->second;
        ++it;
    }
        
    for_(beginIdx, endIdx,
         [=, &qstates, &func](QstateIdxType idx) {
             V prob = V(1.);
             for (int qstatesIdx = 0; (int)qstatesIdx < (int)nQubitStates; ++qstatesIdx) {
                 const Complex &state = qstates[qstatesIdx]->getStateByGlobalIdx(idx);
                 prob *= func(state);
             }
             values[idx] = prob;
         });
    delete[] qstates;
}



void CPUQubits::getStates(Complex *states,
                          QstateIdxType beginIdx, QstateIdxType endIdx) const {
    
    getValues(states, beginIdx, endIdx, null<real>);
}

void CPUQubits::getProbabilities(real *probArray,
                                 QstateIdxType beginIdx, QstateIdxType endIdx) const {

    getValues(probArray, beginIdx, endIdx, abs2<real>);
}



CPUQubitStates::CPUQubitStates() {
    qstates_ = NULL;
}

CPUQubitStates::~CPUQubitStates() {
    deallocate();
}



void CPUQubitStates::allocate(const IdList &qregIdList) {
    qregIdList_ = qregIdList;
    assert(qstates_ == NULL);
    nStates_ = One << qregIdList_.size();
    qstates_ = (Complex*)malloc(sizeof(Complex) * nStates_);
    reset();
}
    
void CPUQubitStates::deallocate() {
    if (qstates_ != NULL)
        free(qstates_);
    qstates_ = NULL;
}

void CPUQubitStates::reset() {
    memset(qstates_, 0, sizeof(Complex) * nStates_);
    qstates_[0] = Complex(1.);
}

int CPUQubitStates::getLane(int qregId) const {
    IdList::const_iterator it = std::find(qregIdList_.begin(), qregIdList_.end(), qregId);
    assert(it != qregIdList_.end());
    return (int)std::distance(qregIdList_.begin(), it);
}

const Complex &CPUQubitStates::getStateByGlobalIdx(QstateIdxType idx) const {
    QstateIdxType localIdx = convertToLocalLaneIdx(idx);
    return qstates_[localIdx];
}

QstateIdxType CPUQubitStates::convertToLocalLaneIdx(QstateIdxType globalIdx) const {
    QstateIdxType localIdx = 0;
    for (int bitPos = 0; bitPos < (int)qregIdList_.size(); ++bitPos) {
        int qregId = qregIdList_[bitPos]; 
        if ((One << qregId) & globalIdx)
            localIdx |= One << bitPos;
    }
    return localIdx;
}



int cpuMeasure(real randNum, CPUQubitStates &qstates, int qregId) {

    int cregValue = -1;
    
    int lane = qstates.getLane(qregId);

    QstateIdxType bitmask_lane = One << lane;
    QstateIdxType bitmask_hi = ~((Two << lane) - 1);
    QstateIdxType bitmask_lo = (One << lane) - 1;
    QstateIdxType nStates = One << (qstates.getNLanes() - 1);
    real prob = real(0.);
    
    prob = sum(0, nStates,
               [=, &qstates](QstateIdxType idx) {
                   QstateIdxType idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                   const Complex &qs = qstates[idx_lo];
                   return abs2<real>(qs);
               });

    if (randNum < prob) {
        cregValue = 0;
        real norm = real(1.) / std::sqrt(prob);

        for_(0, nStates,
             [=, &qstates](QstateIdxType idx) {
                 QstateIdxType idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                 QstateIdxType idx_hi = idx_lo | bitmask_lane;
                 qstates[idx_lo] *= norm;
                 qstates[idx_hi] = real(0.);
             });
    }
    else {
        cregValue = 1;
        real norm = real(1.) / std::sqrt(real(1.) - prob);
        for_(0, nStates,
             [=, &qstates](QstateIdxType idx) {
                 QstateIdxType idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
                 QstateIdxType idx_hi = idx_lo | bitmask_lane;
                 qstates[idx_lo] = real(0.);
                 qstates[idx_hi] *= norm;
             });
        
    }

    return cregValue;
}
    
void cpuApplyReset(CPUQubitStates &qstates, int qregId) {

    int lane = qstates.getLane(qregId);

    QstateIdxType bitmask_lane = One << lane;
    QstateIdxType bitmask_hi = ~((Two << lane) - 1);
    QstateIdxType bitmask_lo = (One << lane) - 1;
    QstateIdxType nStates = One << (qstates.getNLanes() - 1);

    /* Assuming reset is able to be applyed after measurement.
     * Ref: https://quantumcomputing.stackexchange.com/questions/3908/possibility-of-a-reset-quantum-gate */
    for_(0, nStates,
         [=, &qstates](QstateIdxType idx) {
             QstateIdxType idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
             QstateIdxType idx_hi = idx_lo | bitmask_lane;
             qstates[idx_lo] = qstates[idx_hi];
             qstates[idx_hi] = real(0.);
         });
}

void cpuApplyUnaryGate(const CMatrix2x2 &mat, CPUQubitStates &qstates, int qregId) {
    
    int lane = qstates.getLane(qregId);
    
    QstateIdxType bitmask_lane = One << lane;
    QstateIdxType bitmask_hi = ~((Two << lane) - 1);
    QstateIdxType bitmask_lo = (One << lane) - 1;
    QstateIdxType nStates = One << (qstates.getNLanes() - 1);
    for (QstateIdxType idx = 0; idx < nStates; ++idx) {
        QstateIdxType idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
        QstateIdxType idx_hi = idx_lo | bitmask_lane;
        const Complex &qs0 = qstates[idx_lo];
        const Complex &qs1 = qstates[idx_hi];
        Complex qsout0 = mat(0, 0) * qs0 + mat(0, 1) * qs1;
        Complex qsout1 = mat(1, 0) * qs0 + mat(1, 1) * qs1;
        qstates[idx_lo] = qsout0;
        qstates[idx_hi] = qsout1;
    }
}

void cpuApplyControlGate(const CMatrix2x2 &mat, CPUQubitStates &qstates, int controlId, int targetId) {

    int lane0 = qstates.getLane(controlId);
    int lane1 = qstates.getLane(targetId);
    QstateIdxType bitmask_control = One << lane0;
    QstateIdxType bitmask_target = One << lane1;
        
    QstateIdxType bitmask_lane_max = std::max(bitmask_control, bitmask_target);
    QstateIdxType bitmask_lane_min = std::min(bitmask_control, bitmask_target);
        
    QstateIdxType bitmask_hi = ~(bitmask_lane_max * 2 - 1);
    QstateIdxType bitmask_mid = (bitmask_lane_max - 1) & ~((bitmask_lane_min << 1) - 1);
    QstateIdxType bitmask_lo = bitmask_lane_min - 1;
        
    QstateIdxType nStates = One << (qstates.getNLanes() - 2);
    for_(0, nStates,
         [=, &qstates](QstateIdxType idx) {
             QstateIdxType idx_0 = ((idx << 2) & bitmask_hi) | ((idx << 1) & bitmask_mid) | (idx & bitmask_lo) | bitmask_control;
             QstateIdxType idx_1 = idx_0 | bitmask_target;
             
             const Complex &qs0 = qstates[idx_0];
             const Complex &qs1 = qstates[idx_1];;
             Complex qsout0 = mat(0, 0) * qs0 + mat(0, 1) * qs1;
             Complex qsout1 = mat(1, 0) * qs0 + mat(1, 1) * qs1;
             qstates[idx_0] = qsout0;
             qstates[idx_1] = qsout1;
         });
}
