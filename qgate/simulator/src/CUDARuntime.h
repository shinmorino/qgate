#pragma once

#include "cudafuncs.h"
#include <map>

namespace cuda_runtime {

class CUDAQubitStates;
class DeviceComplex;

struct DeviceQubitStates {
    __host__
    DeviceQubitStates() : d_qregIdList_(NULL), nQregIds_(-1), d_qstates_(NULL) { }

    __host__
    void allocate(const IdList &qregIdList);

    __host__
    void deallocate();

    __host__
    void reset();
    
    int *d_qregIdList_;
    int nQregIds_;
    DeviceComplex *d_qstates_;
    QstateIdxType nStates_;
};



class CUDAQubits {
public:

    CUDAQubits();

    ~CUDAQubits();

    void setQregIdList(const IdList &qregIdList);
    
    void allocateQubitStates(int key, const IdList &qregIdList);

    void deallocate();

    void prepare();
    
    CUDAQubitStates &operator[](int key);

    const CUDAQubitStates &operator[](int key) const;

    QstateIdxType getListSize() const;
    
    void getStates(Complex *states,
                   QstateIdxType beginIdx, QstateIdxType endIdx) const;
    
    void getProbabilities(real *prob,
                          QstateIdxType beginIdx, QstateIdxType endIdx) const;
    
    
    template<class V, class F>
    void getValues(V *buf, QstateIdxType beginIdx, QstateIdxType endIdx, const F &func) const;
    
private:
    void freeDeviceBuffer();
    
    IdList qregIdList_;
    typedef std::map<int, CUDAQubitStates*> CUDAQubitStatesMap;
    CUDAQubitStatesMap cuQubitStatesMap_;

    DeviceQubitStates *d_devQubitStatesArray_;
    
    /* hidden copy ctor */
    CUDAQubits(const CUDAQubits &);
};

    
/* representing entangled qubits, or a single qubit or entangled qubits. */
class CUDAQubitStates {
public:
    CUDAQubitStates();

    ~CUDAQubitStates();
    
    void allocate(const IdList &qregIdList);
    
    void deallocate();

    void reset();
    
    size_t getNLanes() const {
        return qregIdList_.size();
    }

    int getLane(int qregId) const;

    DeviceComplex *getDevicePtr();

    const DeviceComplex *getDevicePtr() const;

    const DeviceQubitStates &getDeviceQubitStates() const {
        return devQstates_;
    }
    
private:
    IdList qregIdList_;
    DeviceQubitStates devQstates_;
    
    /* hidden copy ctor */
    CUDAQubitStates(const CUDAQubitStates &);
};



class CUDARuntime {
public:
    CUDARuntime() {
        cuQubits_ = NULL;
    }

    ~CUDARuntime() { }

    void setQubits(CUDAQubits *d_qubits) {
        cuQubits_ = d_qubits;
    }
    
    void setAllQregIds(const IdList &qregIdList);
    
    void allocateQubitStates(int circuit_idx, const IdList &qregset);

    const CUDAQubits &getQubits() const {
        return *cuQubits_;
    }
    
    int measure(real randNum, int key, int qregId);
    
    void applyReset(int key, int qregId);

    void applyUnaryGate(const CMatrix2x2 &mat, int key, int qregId);

    void applyControlGate(const CMatrix2x2 &mat, int key, int controlId, int targetId);

private:
    CUDAQubits *cuQubits_;
};


}
