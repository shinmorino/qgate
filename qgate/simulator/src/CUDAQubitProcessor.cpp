#include "CUDAQubitProcessor.h"
#include "CUDAQubitStates.h"
#include "DeviceProcPrimitives.h"
#include "DeviceTypes.h"
#include "DeviceGetStates.h"
#include "Parallel.h"
#include <algorithm>
#include <numeric>

using namespace qgate_cuda;
using qgate::QstateIdx;
using qgate::QstateSize;


namespace {

template<class R>
struct abs2 {
    __device__ __forceinline__
    R operator()(const DeviceComplexType<R> &c) const {
        return c.real * c.real + c.imag * c.imag;
    }
};

template<class V>
struct null {
    __device__ __forceinline__
    const DeviceComplexType<V> &operator()(const DeviceComplexType<V> &c) const {
        return c;
    }
};

template<class V> struct DeviceType;
template<> struct DeviceType<float> { typedef float Type; };
template<> struct DeviceType<double> { typedef double Type; };
template<> struct DeviceType<qgate::ComplexType<float>> { typedef DeviceComplexType<float> Type; };
template<> struct DeviceType<qgate::ComplexType<double>> { typedef DeviceComplexType<double> Type; };
}


using qgate::Qone;
using qgate::Qtwo;


template<class real>
CUDAQubitProcessor<real>::CUDAQubitProcessor(CUDADevices &devices) : devices_(devices) { }

template<class real>
CUDAQubitProcessor<real>::~CUDAQubitProcessor() { }

template<class real>
void CUDAQubitProcessor<real>::clear() {
    for (auto &proc : procs_)
        delete proc;
    procs_.clear();
}

template<class real>
void CUDAQubitProcessor<real>::prepare() {
}

template<class real>
void CUDAQubitProcessor<real>::synchronize() {
    for (auto & device : activeDevices_)
        device->synchronize();
}

/* synchronize all active devices */
template<class real>
void CUDAQubitProcessor<real>::synchronizeMultiDevice() {
    if (activeDevices_.size() != 1)
        synchronize();
}


template<class real> void CUDAQubitProcessor<real>::
initializeQubitStates(const qgate::IdList &qregIdList, qgate::QubitStates &qstates,
                      int nLanesPerChunk, qgate::IdList &_deviceIds) {
    CUQStates &cuQstates = static_cast<CUQStates&>(qstates);

    qgate::IdList deviceIds = _deviceIds;
    if (deviceIds.empty()) {
        for (int idx = 0; idx < devices_.size(); ++idx)
            deviceIds.push_back(idx);
    }
    
    int nQregIds = (int)qregIdList.size();
    if (nLanesPerChunk == -1)
        nLanesPerChunk = devices_.maxNLanesInDevice();

    int nRequiredChunks;
    if (nLanesPerChunk < nQregIds)
        nRequiredChunks = 1 << (nQregIds - nLanesPerChunk);
    else
        nRequiredChunks = 1;

    if ((int)deviceIds.size() < nRequiredChunks) {
        throwError("Number of devices is not enough, required = %d, current = %d.",
                   nRequiredChunks, devices_.size());
    }
    if (nRequiredChunks == 1)
        nLanesPerChunk = (int)qregIdList.size();
    
    try {
        CUDADeviceList memoryOwners;
        for (int iChunk = 0; iChunk < nRequiredChunks; ++iChunk) {
            CUDADevice &device = devices_[deviceIds[iChunk]];
            memoryOwners.push_back(&device);
            procs_.push_back(new DeviceProcPrimitives<real>(device));
            activeDevices_.push_back(&device);
        }
        cuQstates.allocate(qregIdList, memoryOwners, nLanesPerChunk);
    }
    catch (...) {
        qstates.deallocate();
        throw;
    }
    /* remove duplicates in activeDevices_. */
    auto lessDeviceNumber = [](const CUDADevice *dev0, const CUDADevice *dev1) {
        return dev0->getDeviceNumber() < dev1->getDeviceNumber();
    };
    std::sort(activeDevices_.begin(), activeDevices_.end(), lessDeviceNumber);
    auto eqDeviceNumber = [](const CUDADevice *dev0, const CUDADevice *dev1) {
        return dev0->getDeviceNumber() == dev1->getDeviceNumber();
    };
    auto duplicates = std::unique(activeDevices_.begin(), activeDevices_.end(), eqDeviceNumber);
    activeDevices_.erase(duplicates, activeDevices_.end());
}

template<class real>
void CUDAQubitProcessor<real>::resetQubitStates(qgate::QubitStates &qstates) {
    CUQStates &cuQstates = static_cast<CUQStates&>(qstates);
    DevicePtr &devPtr = cuQstates.getDevicePtr();
    auto resetFunc = [&](int chunkIdx, QstateIdx begin, QstateIdx end) {
                         procs_[chunkIdx]->fillZero(devPtr, begin, end);
                     };
    dispatchToDevices(cuQstates, resetFunc);

    Complex cOne(1.);
    procs_[0]->set(devPtr, &cOne, 0, sizeof(cOne));
    synchronizeMultiDevice();
}

template<class real>
int CUDAQubitProcessor<real>::measure(double randNum,
                                      qgate::QubitStates &qstates, int qregId) {

    CUQStates &cuQstates = static_cast<CUQStates&>(qstates);
    DevicePtr &devPtr = cuQstates.getDevicePtr();
    
    int cregValue = -1;

    int lane = cuQstates.getLane(qregId);

    auto calcProbLaunch = [&](int chunkIdx, QstateIdx begin, QstateIdx end) {
                              procs_[chunkIdx]->calcProb_launch(devPtr, lane, begin, end);
                          };
    apply(lane, cuQstates, calcProbLaunch);

    std::vector<real> partialSum(procs_.size());
    auto calcProbSync = [&](int chunkIdx) {
                            partialSum[chunkIdx] = procs_[chunkIdx]->calcProb_sync();
                        };
    qgate::Parallel(cuQstates.getNumChunks()).run(calcProbSync);
    
    real prob = std::accumulate(partialSum.begin(), partialSum.end(), real(0.));

    /* reset bits */
    if (real(randNum) < prob) {
        cregValue = 0;
        auto set_0 = [&](int chunkIdx, QstateIdx begin, QstateIdx end){
                         procs_[chunkIdx]->measure_set0(devPtr, lane, prob, begin, end);
                     };
        apply(lane, cuQstates, set_0);
    }
    else {
        cregValue = 1;
        auto set_1 = [&](int chunkIdx, QstateIdx begin, QstateIdx end){
                         procs_[chunkIdx]->measure_set1(devPtr, lane, prob, begin, end);
                     };
        apply(lane, cuQstates, set_1);
    }
    synchronizeMultiDevice();

    return cregValue;
}


template<class real>
void CUDAQubitProcessor<real>::applyReset(qgate::QubitStates &qstates, int qregId) {
    
    CUQStates &cuQstates = static_cast<CUQStates&>(qstates);
    DevicePtr &devPtr = cuQstates.getDevicePtr();
    
    int lane = cuQstates.getLane(qregId);
    
    auto reset = [&](int chunkIdx, QstateIdx begin, QstateIdx end) {
                     procs_[chunkIdx]->applyReset(devPtr, lane, begin, end);
                 };
    apply(lane, cuQstates, reset);
    synchronizeMultiDevice();
}


template<class real>
void CUDAQubitProcessor<real>::applyUnaryGate(const Matrix2x2C64 &mat, qgate::QubitStates &qstates, int qregId) {
    CUQStates &cuQstates = static_cast<CUQStates&>(qstates);
    DevicePtr &devPtr = cuQstates.getDevicePtr();

    DeviceMatrix2x2C<real> dmat(mat);

    int lane = cuQstates.getLane(qregId);
    auto applyUnaryGate = [&](int chunkIdx, QstateIdx begin, QstateIdx end) {
                              procs_[chunkIdx]->applyUnaryGate(dmat, devPtr, lane, begin, end);
                          };
    apply(lane, cuQstates, applyUnaryGate);
    synchronizeMultiDevice();
}


template<class real>
void CUDAQubitProcessor<real>::applyControlGate(const Matrix2x2C64 &mat, QubitStates &qstates,
                                                int controlId, int targetId) {
    CUQStates &cuQstates = static_cast<CUQStates&>(qstates);
    DevicePtr &devPtr = cuQstates.getDevicePtr();

    DeviceMatrix2x2C<real> dmat(mat);
    int controlLane = cuQstates.getLane(controlId);
    int targetLane = cuQstates.getLane(targetId);

    auto applyControlGate = [&](int chunkIdx, QstateIdx begin, QstateIdx end) {
                                procs_[chunkIdx]->applyControlGate(dmat, devPtr, controlLane, targetLane,
                                                                   begin, end);
                            };
    applyHi(controlLane, cuQstates, applyControlGate);

    synchronizeMultiDevice();
}


template<class real> template<class F> void
CUDAQubitProcessor<real>::dispatchToDevices(CUQStates &cuQstates, const F &f) {
    int nLanes = cuQstates.getNQregs();
    QstateSize nThreads = Qone << nLanes;
    QstateSize nThreadsPerDevice = nThreads / cuQstates.getNumChunks();
    dispatchToDevices(cuQstates, f, 0, nThreads, nThreadsPerDevice);
}

template<class real> template<class F> void
CUDAQubitProcessor<real>::dispatchToDevices(CUQStates &cuQstates, const F &f, QstateIdx begin, QstateIdx end, QstateSize nThreadsPerDevice) {
    int nChunks = (int)cuQstates.getNumChunks();
    for (int iChunk = 0; iChunk < nChunks; ++iChunk) {
        QstateIdx beginInChunk = std::max(nThreadsPerDevice * iChunk, begin);
        QstateIdx endInChunk = std::min(nThreadsPerDevice * (iChunk + 1), end);
        if (beginInChunk != endInChunk)
            f(iChunk, beginInChunk, endInChunk);
    }
}

template<class real> template<class F> void CUDAQubitProcessor<real>::
apply(const qgate::IdList &lanes, CUQStates &cuQstates, F &f) {

    int nLanes = cuQstates.getNQregs();
    int nLanesInDevice = cuQstates.getNLanesInDevice();
    qgate::IdList bits;
    for (int idx = 0; idx < (int)lanes.size(); ++idx) {
        int lane = lanes[idx];
        if (nLanesInDevice <= lane) {
            int bit = 1 << (lane - nLanesInDevice);
            bits.push_back(bit);
        }
    }

    qgate::IdList ordered;
    
    int nChunksPerGroup = 1 << bits.size();
	int nChunks = cuQstates.getNumChunks();
	int nGroups = nChunks / nChunksPerGroup;
    
    if (nChunksPerGroup == 1) {
        for (int idx = 0; idx < nGroups; ++idx)
            ordered.push_back(idx);
    }
    else {
        for (int iGroup = 0; iGroup < nGroups; ++iGroup) {
            /* calculate base idx */
            int nBits = (int)bits.size();
            int mask = bits[0]- 1; /* mask_lo */
            int idx_base = iGroup | mask;
            for (int iBit = 0; iBit < nBits - 1; ++iBit) {
                int mask_lo = bits[iBit] - 1;
                int mask_hi = ~((bits[iBit + 1] << 1) - 1);
                idx_base |= (iGroup << iBit) & (mask_lo & mask_hi);
            }
            mask = ~((2 << bits.back()) - 1);
            idx_base |= (iGroup << (nBits - 1)) | mask;
            
            for (int idx = 0; idx < nChunksPerGroup; ++idx) {
                int chunkIdx = idx_base;
                for (int iBit = 0; iBit < nBits; ++iBit) {
                    if (idx & iBit)
                        chunkIdx |= bits[iBit];
                }
                ordered.push_back(chunkIdx);
            }
        }
    }
    
    int nInputs = 1 << (int)lanes.size();
    QstateSize nThreads = (Qone << nLanes) / (1 << lanes.size());
    QstateSize nThreadsPerChunk = nThreads / nChunks;
    for (int iChunk = 0; iChunk < nChunks; ++iChunk) {
        QstateIdx begin = nThreadsPerChunk * iChunk;
        QstateIdx end = nThreadsPerChunk * (iChunk + 1);
        f(ordered[iChunk], begin, end);
    }
}

template<class real> template<class F> void CUDAQubitProcessor<real>::
apply(int bitPos, CUQStates &cuQstates, const F &f) {
    apply(bitPos, cuQstates, f, true, true);
}


template<class real> template<class F> void CUDAQubitProcessor<real>::
applyHi(int bitPos, CUQStates &cuQstates, const F &f) {
    apply(bitPos, cuQstates, f, true, false);
}

template<class real> template<class F> void CUDAQubitProcessor<real>::
applyLo(int bitPos, CUQStates &cuQstates, const F &f) {
    apply(bitPos, cuQstates, f, false, true);
}

template<class real> template<class F> void
CUDAQubitProcessor<real>::apply(int bitPos, CUQStates &cuQstates, const F &f,
                                bool runHi, bool runLo) {
    /* 1-bit gate */
    int nLanesInChunk = cuQstates.getNLanesInChunk();
    int nChunks = cuQstates.getNumChunks();
    int nGroups;
    int bit;
    if (nLanesInChunk <= bitPos) {
        bit = 1 << (bitPos - nLanesInChunk);
        nGroups = nChunks / 2;
    }
    else {
        nGroups = nChunks;
        bit = 0;
    }

    qgate::IdList ordered;
    
    int nChunksPerGroup = nChunks / nGroups;
    
    if (nChunksPerGroup == 1) {
        for (int idx = 0; idx < nChunks; ++idx)
            ordered.push_back(idx);
    }
    else {
        for (int idx = 0; idx < nChunksPerGroup; ++idx) {
            /* calculate base idx */
            int mask_0 = bit - 1;
            int mask_1 = ~((bit << 1) - 1);
            int idx_lo = ((idx << 1) | mask_1) & (idx & mask_0);
            if (runLo)
                ordered.push_back(idx_lo);
            int idx_hi = idx_lo | bit;
            if (runHi)
                ordered.push_back(idx_hi);
        }
    }

    QstateSize nThreads = (Qone << cuQstates.getNQregs()) / 2;
    if (runHi != runLo)
        nThreads /= 2;

    QstateSize nThreadsPerChunk = nThreads / (int)ordered.size();
    for (int iChunk = 0; iChunk < ordered.size(); ++iChunk) {
        QstateIdx begin = nThreadsPerChunk * iChunk;
        QstateIdx end = nThreadsPerChunk * (iChunk + 1);
        f(ordered[iChunk], begin, end);
    }
}

/* get states */

template<class real>
void CUDAQubitProcessor<real>::getStates(void *array, QstateIdx arrayOffset,
                                         MathOp op,
                                         const QubitStatesList &qstatesList,
                                         QstateIdx beginIdx, QstateIdx endIdx) {

    for (int idx = 0; idx < (int)qstatesList.size(); ++idx) {
        const qgate::QubitStates *qstates = qstatesList[idx];
        if (sizeof(real) == sizeof(float)) {
            abortIf(qstates->getPrec() != qgate::precFP32, "Wrong type");
        }
        else if (sizeof(real) == sizeof(double)) {
            abortIf(qstates->getPrec() != qgate::precFP64, "Wrong type");
        }
    }

    DeviceGetStates<real> getStates(qstatesList, activeDevices_);
    getStates.run(array, arrayOffset, op, beginIdx, endIdx);
}


template class CUDAQubitProcessor<float>;
template class CUDAQubitProcessor<double>;
