#include "CPURuntime.h"
#include <string.h>
#include <algorithm>


namespace {

const QstateIdxType One = 1;
const QstateIdxType Two = 2;

}


Qubits::~Qubits() {
    deallocate();
}


void Qubits::setQregIdList(const IdList &qregIdList) {
    qregIdList_ = qregIdList;
}


void Qubits::allocateQubitStates(int key, const IdList &qregIdList) {
    QubitStates *qstates = new QubitStates();
    qstates->allocate(qregIdList);
    qubitStatesMap_[key] = qstates;
}

void Qubits::deallocate() {
    for (QubitStatesMap::iterator it = qubitStatesMap_.begin();
         it != qubitStatesMap_.end(); ++it) {
        delete it->second;
    }
    qubitStatesMap_.clear();
}
    
QubitStates &Qubits::operator[](int key) {
    QubitStatesMap::iterator it = qubitStatesMap_.find(key);
    assert(it != qubitStatesMap_.end());
    return *it->second;
}

const QubitStates &Qubits::operator[](int key) const {
    QubitStatesMap::const_iterator it = qubitStatesMap_.find(key);
    assert(it != qubitStatesMap_.end());
    return *it->second;
}

QstateIdxType Qubits::getListSize() const {
    return One << qubitStatesMap_.size();
}

namespace {


template<class R>
inline R abs2(const std::complex<R> &c) {
    return c.real() * c.real() + c.imag() + c.imag();
}

template<class R>
inline std::complex<R> null(const std::complex<R> &c) {
    return c;
}


}


template<class V, class F>
void Qubits::getValues(V *values,
                       QstateIdxType beginIdx, QstateIdxType endIdx,
                       const F &func) const {
    
    int nQubitStates = qubitStatesMap_.size();
    QubitStates *qstates[nQubitStates];
    QubitStatesMap::const_iterator it = qubitStatesMap_.begin();
    for (int qstatesIdx = 0; (int)qstatesIdx < (int)qubitStatesMap_.size(); ++qstatesIdx) {
        qstates[qstatesIdx] = it->second;
        ++it;
    }
        
    for (QstateIdxType idx = beginIdx; idx < endIdx; ++idx) {
        V prob = V(1.);
        for (int qstatesIdx = 0; (int)qstatesIdx < (int)nQubitStates; ++qstatesIdx) {
            const Complex &state = qstates[qstatesIdx]->getStateByGlobalIdx(idx);
            prob *= func(state);
        }
        values[idx] = prob;
    }
}



void Qubits::getStates(Complex *states,
                       QstateIdxType beginIdx, QstateIdxType endIdx) const {

    getValues(states, beginIdx, endIdx, null<real>);
}

void Qubits::getProbabilities(real *probArray,
                              QstateIdxType beginIdx, QstateIdxType endIdx) const {

    getValues(probArray, beginIdx, endIdx, abs2<real>);
}



QubitStates::QubitStates() {
    qstates_ = NULL;
}

QubitStates::~QubitStates() {
    deallocate();
}



void QubitStates::allocate(const IdList &qregIdList) {
    qregIdList_ = qregIdList;
    assert(qstates_ == NULL);
    nStates_ = One << qregIdList_.size();
    qstates_ = (Complex*)malloc(sizeof(Complex) * nStates_);
    reset();
}
    
void QubitStates::deallocate() {
    if (qstates_ != NULL)
        free(qstates_);
    qstates_ = NULL;
}

void QubitStates::reset() {
    memset(qstates_, 0, sizeof(Complex) * nStates_);
    qstates_[0] = Complex(1.);
}

int QubitStates::getLane(int qregId) const {
    IdList::const_iterator it = std::find(qregIdList_.begin(), qregIdList_.end(), qregId);
    assert(it != qregIdList_.end());
    return std::distance(qregIdList_.begin(), it);
}

const Complex &QubitStates::getStateByGlobalIdx(QstateIdxType idx) const {
    QstateIdxType localIdx = convertToLocalLaneIdx(idx);
    return qstates_[localIdx];
}

QstateIdxType QubitStates::convertToLocalLaneIdx(QstateIdxType globalIdx) const {
    QstateIdxType localIdx = 0;
    for (int bitPos = 0; bitPos < (int)qregIdList_.size(); ++bitPos) {
        int qregId = qregIdList_[bitPos]; 
        if ((One << qregId) & globalIdx)
            localIdx |= One << bitPos;
    }
    return localIdx;
}



void CPURuntime::setAllQregIds(const IdList &qregIdList) {
    qubits_->setQregIdList(qregIdList);
}
    
void CPURuntime::allocateQubitStates(int key, const IdList &qregset) {
    qubits_->allocateQubitStates(key, qregset);
}

int CPURuntime::measure(real randNum, int key, int qregId) {
    QubitStates &qstates = (*qubits_)[key];

    int cregValue = -1;
    
    int lane = qstates.getLane(qregId);

    QstateIdxType bitmask_lane = One << lane;
    QstateIdxType bitmask_hi = ~((Two << lane) - 1);
    QstateIdxType bitmask_lo = (One << lane) - 1;
    QstateIdxType nStates = One << (qstates.getNLanes() - 1);
    real prob = real(0.);
    for (QstateIdxType idx = 0; idx < nStates; ++idx) {
        QstateIdxType idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
        const Complex &qs = qstates[idx_lo];
        prob += abs2(qs);
    }

    if (randNum < prob) {
        cregValue = 0;
        real norm = real(1.) / std::sqrt(prob);
        for (QstateIdxType idx = 0; idx < nStates; ++idx) {
            QstateIdxType idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
            QstateIdxType idx_hi = idx_lo | bitmask_lane;
            qstates[idx_lo] *= norm;
            qstates[idx_hi] = real(0.);
        }
    }
    else {
        cregValue = 1;
        real norm = 1. / std::sqrt(real(1.) - prob);
        for (QstateIdxType idx = 0; idx < nStates; ++idx) {
            QstateIdxType idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
            QstateIdxType idx_hi = idx_lo | bitmask_lane;
            qstates[idx_lo] = real(0.);
            qstates[idx_hi] *= norm;
        }
    }

    return cregValue;
}
    
void CPURuntime::applyReset(int key, int qregId) {
    QubitStates &qstates = (*qubits_)[key];
    
    int lane = qstates.getLane(qregId);

    QstateIdxType bitmask_lane = One << lane;
    QstateIdxType bitmask_hi = ~((Two << lane) - 1);
    QstateIdxType bitmask_lo = (One << lane) - 1;
    QstateIdxType nStates = One << (qstates.getNLanes() - 1);

    /* Assuming reset is able to be applyed after measurement.
     * Ref: https://quantumcomputing.stackexchange.com/questions/3908/possibility-of-a-reset-quantum-gate */
    for (QstateIdxType idx = 0; idx < nStates; ++idx) {
        QstateIdxType idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
        QstateIdxType idx_hi = idx_lo | bitmask_lane;
        qstates[idx_lo] = qstates[idx_hi];
        qstates[idx_hi] = real(0.);
    }
}

void CPURuntime::applyUnaryGate(const CMatrix2x2 &mat, int key, int qregId) {
    QubitStates &qstates = (*qubits_)[key];
    
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

void CPURuntime::applyControlGate(const CMatrix2x2 &mat, int key, int controlId, int targetId) {
    QubitStates &qstates = (*qubits_)[key];
    
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
    for (QstateIdxType idx = 0; idx < nStates; ++idx) {
        QstateIdxType idx_0 = ((idx << 2) & bitmask_hi) | ((idx << 1) & bitmask_mid) | (idx & bitmask_lo) | bitmask_control;
        QstateIdxType idx_1 = idx_0 | bitmask_target;
            
        const Complex &qs0 = qstates[idx_0];
        const Complex &qs1 = qstates[idx_1];;
        Complex qsout0 = mat(0, 0) * qs0 + mat(0, 1) * qs1;
        Complex qsout1 = mat(1, 0) * qs0 + mat(1, 1) * qs1;
        qstates[idx_0] = qsout0;
        qstates[idx_1] = qsout1;
    }
}
