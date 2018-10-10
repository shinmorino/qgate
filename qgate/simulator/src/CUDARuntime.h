#pragma once

#include <map>
#include "DeviceTypes.h"
#include "DeviceSum.h"

namespace cuda_runtime {


class CUDAQubitStates;
struct CUDARuntimeResource;

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
    
    void addQubitStates(int key, CUDAQubitStates *qstates);

    void detachQubitStates();
    
    CUDAQubitStates &operator[](int key);

    const CUDAQubitStates &operator[](int key) const;

    QstateIdxType getNStates() const; 

    void prepare();
   
    void getStates(Complex *states,
                   QstateIdxType beginIdx, QstateIdxType endIdx,
                   CUDARuntimeResource &rsrc) const;
    
    void getProbabilities(real *prob,
                          QstateIdxType beginIdx, QstateIdxType endIdx,
                          CUDARuntimeResource &rsrc) const;

    /* public to enable device lambda. */
    template<class V, class F>
    void getValues(V *values,
                   QstateIdxType beginIdx, QstateIdxType endIdx,
                   const F &func, CUDARuntimeResource &rsrc) const;
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
    
    int getNLanes() const {
        return (int)qregIdList_.size();
    }

    int getLane(int qregId) const;

    DeviceComplex *getDevicePtr() {
        return devQstates_.d_qstates_;
    }

    const DeviceComplex *getDevicePtr() const {
        return devQstates_.d_qstates_;
    }

    const DeviceQubitStates &getDeviceQubitStates() const {
        return devQstates_;
    }
    
private:
    IdList qregIdList_;
    DeviceQubitStates devQstates_;
    
    /* hidden copy ctor */
    CUDAQubitStates(const CUDAQubitStates &);
};


struct CUDARuntimeResource {
    enum { hMemBufSize = 1 << 28 };

    CUDARuntimeResource() {
        h_buffer_ = NULL;
    }
    ~CUDARuntimeResource() {
        finalize();
    }
    
    void prepare() {
        deviceSum_.prepare();
        throwOnError(cudaHostAlloc(&h_buffer_, 1 << 28, cudaHostAllocPortable));
    }
    
    void finalize() {
        deviceSum_.finalize();
        if (h_buffer_ != NULL)
            throwOnError(cudaFreeHost(h_buffer_));
        h_buffer_ = NULL;
    }

    template<class V>
    V *getHostMem() {
        return static_cast<V*>(h_buffer_);
    }

    template<class V>
    size_t hostMemSize() const {
        return (size_t)hMemBufSize / sizeof(V);
    }
    
    DeviceSum deviceSum_;
    void *h_buffer_;
};

    
int cudaMeasure(real randNum, CUDAQubitStates &qstates, int qregId, CUDARuntimeResource &rsrc);

void cudaApplyReset(CUDAQubitStates &qstates, int qregId);

void cudaApplyUnaryGate(const CMatrix2x2 &mat, CUDAQubitStates &qstates, int qregId);

void cudaApplyControlGate(const CMatrix2x2 &mat, CUDAQubitStates &qstates, int controlId, int targetId);

}
