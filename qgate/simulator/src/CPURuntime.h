#pragma once

#include "Types.h"
#include <map>


class CPUQubitStates;


class CPUQubits {
public:

    CPUQubits() { }

    ~CPUQubits();
    
    void addQubitStates(int key, CPUQubitStates *qstates);

    void detachQubitStates();
    
    CPUQubitStates &operator[](int key);

    const CPUQubitStates &operator[](int key) const;

    QstateIdxType getNStates() const;
    
    void getStates(Complex *states,
                   QstateIdxType beginIdx, QstateIdxType endIdx) const;
    
    void getProbabilities(real *prob,
                          QstateIdxType beginIdx, QstateIdxType endIdx) const;
    
private:
    template<class V, class F>
    void getValues(V *buf, QstateIdxType beginIdx, QstateIdxType endIdx, const F &func) const;
    
    IdList qregIdList_;
    typedef std::map<int, CPUQubitStates*> CPUQubitStatesMap;
    CPUQubitStatesMap cpuQubitStatesMap_;


    /* hidden copy ctor */
    CPUQubits(const CPUQubits &);
};


    
/* representing entangled qubits, or a single qubit or entangled qubits. */
class CPUQubitStates {
public:
    CPUQubitStates();

    ~CPUQubitStates();
    
    void allocate(const IdList &qregIdList);
    
    void deallocate();

    void reset();

    int getNLanes() const {
        return (int)qregIdList_.size();
    }

    int getLane(int qregId) const;

    Complex &operator[](QstateIdxType idx) {
        return qstates_[idx];
    }
    
    const Complex &operator[](QstateIdxType idx) const {
        return qstates_[idx];
    }

    const Complex &getStateByGlobalIdx(QstateIdxType idx) const;

    QstateIdxType convertToLocalLaneIdx(QstateIdxType idx) const;
    
private:
    QstateIdxType nStates_;
    IdList qregIdList_;
    Complex *qstates_;
    
    /* hidden copy ctor */
    CPUQubitStates(const CPUQubitStates &);
};


    
int cpuMeasure(real randNum, CPUQubitStates &qstates, int qregId);

void cpuApplyReset(CPUQubitStates &qstates, int qregId);

void cpuApplyUnaryGate(const CMatrix2x2 &mat, CPUQubitStates &qstates, int qregId);

void cpuApplyControlGate(const CMatrix2x2 &mat, CPUQubitStates &qstates, int controlId, int targetId);

