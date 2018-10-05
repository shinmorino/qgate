#pragma once

#include "Types.h"
#include <map>


class QubitStates;


class Qubits {
public:

    Qubits() { }

    ~Qubits();

    void setQregIdList(const IdList &qregIdList);
    
    void allocateQubitStates(int key, const IdList &qregIdList);

    void deallocate();
    
    QubitStates &operator[](int key);

    const QubitStates &operator[](int key) const;

    QstateIdxType getListSize() const;
    
    void getProbabilities(real *prob,
                          QstateIdxType beginIdx, QstateIdxType endIdx) const;
    
private:
    IdList qregIdList_;
    typedef std::map<int, QubitStates*> QubitStatesMap;
    QubitStatesMap qubitStatesMap_;
};


    
/* representing entangled qubits, or a single qubit or entangled qubits. */
class QubitStates {
public:
    QubitStates();

    ~QubitStates();
    
    void allocate(const IdList &qregIdList);
    
    void deallocate();

    void reset();

    size_t getNLanes() const {
        return qregIdList_.size();
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
};



class CPURuntime {
public:
    CPURuntime() {
        qubits_ = NULL;
    }

    ~CPURuntime() { }

    void setQubits(Qubits *qubits) {
        qubits_ = qubits;
    }
    
    void setAllQregIds(const IdList &qregIdList);
    
    void allocateQubitStates(int circuit_idx, const IdList &qregset);

    const Qubits &getQubits() const {
        return *qubits_;
    }
    
    int measure(real randNum, int key, int qregId);
    
    void applyReset(int key, int qregId);

    void applyUnaryGate(const CMatrix2x2 &mat, int key, int qregId);

    void applyControlGate(const CMatrix2x2 &mat, int key, int controlId, int targetId);

private:
    Qubits *qubits_;
};
