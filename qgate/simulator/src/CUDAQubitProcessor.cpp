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
initializeQubitStates(qgate::QubitStates &qstates,
                      int nLanes, int nLanesPerChunk, qgate::IdList &_deviceIds) {
    CUQStates &cuQstates = static_cast<CUQStates&>(qstates);

    qgate::IdList deviceIds = _deviceIds;
    if (deviceIds.empty()) {
        for (int idx = 0; idx < devices_.size(); ++idx)
            deviceIds.push_back(idx);
    }
    
    if (nLanesPerChunk == -1)
        nLanesPerChunk = devices_.maxNLanesInDevice();

    int nRequiredChunks;
    if (nLanesPerChunk < nLanes)
        nRequiredChunks = 1 << (nLanes - nLanesPerChunk);
    else
        nRequiredChunks = 1;

    if ((int)deviceIds.size() < nRequiredChunks) {
        throwError("Number of devices is not enough, required = %d, current = %d.",
                   nRequiredChunks, devices_.size());
    }
    if (nRequiredChunks == 1)
        nLanesPerChunk = nLanes;
    
    try {
        CUDADeviceList memoryOwners;
        for (int iChunk = 0; iChunk < nRequiredChunks; ++iChunk) {
            CUDADevice &device = devices_[deviceIds[iChunk]];
            memoryOwners.push_back(&device);
            procs_.push_back(new DeviceProcPrimitives<real>(device));
            activeDevices_.push_back(&device);
        }
        cuQstates.allocate(memoryOwners, nLanes, nLanesPerChunk);
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
    QstateSize nStates = Qone << cuQstates.getNLanes();
    qgate::IdList allChunks;
    for (int idx = 0; idx < (int)cuQstates.getNumChunks(); ++idx)
        allChunks.push_back(idx);
    dispatch(allChunks, resetFunc, 0, nStates);

    Complex cOne(1.);
    procs_[0]->set(devPtr, &cOne, 0, sizeof(cOne));
    synchronizeMultiDevice();
}

template<class real>
double CUDAQubitProcessor<real>::calcProbability(const qgate::QubitStates &qstates, int localLane) {
    const CUQStates &cuQstates = static_cast<const CUQStates&>(qstates);
    return _calcProbability(cuQstates, localLane);
}

template<class real>
real CUDAQubitProcessor<real>::_calcProbability(const CUDAQubitStates<real> &cuQstates,
                                                int localLane) {
    const DevicePtr &devPtr = cuQstates.getDevicePtr();
    auto calcProbLaunch = [&](int chunkIdx, QstateIdx begin, QstateIdx end) {
        procs_[chunkIdx]->calcProb_launch(devPtr, localLane, begin, end);
    };

    qgate::IdList ordered = orderChunks(localLane, cuQstates, false, true);
    QstateSize nStates = Qone << cuQstates.getNLanes();
    dispatch(ordered, calcProbLaunch, 0, nStates / 2);
    
    std::vector<real> partialSum(ordered.size());
    auto calcProbSync = [&](int chunkIdx) {
                            partialSum[chunkIdx] = procs_[chunkIdx]->calcProb_sync();
                        };
    qgate::Parallel((int)ordered.size()).run(calcProbSync);
    
    real prob = std::accumulate(partialSum.begin(), partialSum.end(), real(0.));
    return prob;
}

template<class real>
int CUDAQubitProcessor<real>::measure(double randNum,
                                      qgate::QubitStates &qstates, int localLane) {

    CUQStates &cuQstates = static_cast<CUQStates&>(qstates);
    real prob = _calcProbability(cuQstates, localLane);

    DevicePtr &devPtr = cuQstates.getDevicePtr();
    int cregValue = -1;

    /* reset bits */
    if (real(randNum) < prob) {
        cregValue = 0;
        auto set_0 = [&](int chunkIdx, QstateIdx begin, QstateIdx end){
                         procs_[chunkIdx]->measure_set0(devPtr, localLane, prob, begin, end);
                     };
        dispatch(localLane, cuQstates, set_0);
    }
    else {
        cregValue = 1;
        auto set_1 = [&](int chunkIdx, QstateIdx begin, QstateIdx end){
                         procs_[chunkIdx]->measure_set1(devPtr, localLane, prob, begin, end);
                     };
        dispatch(localLane, cuQstates, set_1);
    }
    synchronizeMultiDevice();

    return cregValue;
}


template<class real>
void CUDAQubitProcessor<real>::applyReset(qgate::QubitStates &qstates, int localLane) {
    
    CUQStates &cuQstates = static_cast<CUQStates&>(qstates);
    DevicePtr &devPtr = cuQstates.getDevicePtr();
    
    auto reset = [&](int chunkIdx, QstateIdx begin, QstateIdx end) {
                     procs_[chunkIdx]->applyReset(devPtr, localLane, begin, end);
                 };
    dispatch(localLane, cuQstates, reset);
    synchronizeMultiDevice();
}


template<class real>
void CUDAQubitProcessor<real>::applyUnaryGate(const Matrix2x2C64 &mat, qgate::QubitStates &qstates, int localLane) {
    CUQStates &cuQstates = static_cast<CUQStates&>(qstates);
    DevicePtr &devPtr = cuQstates.getDevicePtr();

    DeviceMatrix2x2C<real> dmat(mat);

    auto applyUnaryGate = [&](int chunkIdx, QstateIdx begin, QstateIdx end) {
                              procs_[chunkIdx]->applyUnaryGate(dmat, devPtr, localLane, begin, end);
                          };
    dispatch(localLane, cuQstates, applyUnaryGate);
    synchronizeMultiDevice();
}


template<class real>
void CUDAQubitProcessor<real>::applyControlGate(const Matrix2x2C64 &mat, QubitStates &qstates,
                                                int localControlLane, int localTargetLane) {
    CUQStates &cuQstates = static_cast<CUQStates&>(qstates);
    DevicePtr &devPtr = cuQstates.getDevicePtr();

    DeviceMatrix2x2C<real> dmat(mat);
    auto applyControlGate = [&](int chunkIdx, QstateIdx begin, QstateIdx end) {
        procs_[chunkIdx]->applyControlGate(dmat, devPtr,
                                           localControlLane, localTargetLane,
                                           begin, end);
    };
    
    qgate::IdList ordred = orderChunks(localControlLane, cuQstates, true, false);
    QstateSize nStates = Qone << cuQstates.getNLanes();
    dispatch(ordred, applyControlGate, 0, nStates / 4);

    synchronizeMultiDevice();
}


template<class real> template<class F> void CUDAQubitProcessor<real>::
dispatch(const qgate::IdList &ordered, const F &f, QstateIdx begin, QstateIdx end) {
    int nChunks = (int)ordered.size();
    QstateSize nThreadsPerDevice = (end - begin) / nChunks;
    for (int iChunk = 0; iChunk < nChunks; ++iChunk) {
        QstateIdx beginInChunk = std::min(begin + nThreadsPerDevice * iChunk, end);
        QstateIdx endInChunk = std::min(begin + nThreadsPerDevice * (iChunk + 1), end);
        if (beginInChunk != endInChunk)
            f(iChunk, beginInChunk, endInChunk);
    }
}

template<class real> template<class F> void CUDAQubitProcessor<real>::
dispatch(const qgate::IdList &lanes, CUQStates &cuQstates, F &f) {

    int nLanes = cuQstates.getNLanes();
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

template<class real> qgate::IdList CUDAQubitProcessor<real>::
orderChunks(int bitPos, const CUQStates &cuQstates, bool useHi, bool useLo) const {

    int nLanesInChunk = cuQstates.getNLanesInChunk();
    int nChunks = (int)cuQstates.getNumChunks();
    int nGroups, bit;
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
        for (int idx = 0; idx < nGroups; ++idx) {
            /* calculate base idx */
            int mask_0 = bit - 1;
            int mask_1 = ~((bit << 1) - 1);
            int idx_lo = ((idx << 1) | mask_1) & (idx & mask_0);
            if (useLo)
                ordered.push_back(idx_lo);
            int idx_hi = idx_lo | bit;
            if (useHi)
                ordered.push_back(idx_hi);
        }
    }
    return ordered;
}

template<class real> template<class F> void CUDAQubitProcessor<real>::
dispatch(int bitPos, CUQStates &cuQstates, const F &f) {
    /* 1-bit gate */
    qgate::IdList ordered = orderChunks(bitPos, cuQstates, true, true);
    QstateSize nThreads = (Qone << cuQstates.getNLanes()) / 2;

    QstateSize nThreadsPerChunk = nThreads / (int)ordered.size();
    for (int iChunk = 0; iChunk < (int)ordered.size(); ++iChunk) {
        QstateIdx begin = nThreadsPerChunk * iChunk;
        QstateIdx end = nThreadsPerChunk * (iChunk + 1);
        f(ordered[iChunk], begin, end);
    }
}

/* get states */

template<class real>
void CUDAQubitProcessor<real>::getStates(void *array, QstateIdx arrayOffset,
                                         MathOp op,
                                         const qgate::IdList *laneTransTables, const QubitStatesList &qstatesList,
                                         QstateIdx nStates, QstateIdx begin, QstateIdx step) {
    
    for (int idx = 0; idx < (int)qstatesList.size(); ++idx) {
        const qgate::QubitStates *qstates = qstatesList[idx];
        if (sizeof(real) == sizeof(float)) {
            abortIf(qstates->getPrec() != qgate::precFP32, "Wrong type");
        }
        else if (sizeof(real) == sizeof(double)) {
            abortIf(qstates->getPrec() != qgate::precFP64, "Wrong type");
        }
    }
    
    DeviceGetStates<real> getStates(laneTransTables, qstatesList, activeDevices_);
    getStates.run(array, arrayOffset, op, nStates, begin, step);
}


template class CUDAQubitProcessor<float>;
template class CUDAQubitProcessor<double>;
