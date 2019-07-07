#include "CUDAQubitProcessor.h"
#include "CUDAQubitStates.h"
#include "DeviceProcPrimitives.h"
#include "DeviceGetStates.h"
#include "DeviceProbArrayCalculator.h"
#include "Parallel.h"
#include "BitPermTable.h"
#include "CUDAGlobals.h"
#include "ProcessorRelocator.h"
#include <algorithm>
#include <numeric>
#include <valarray>

using namespace qgate_cuda;
using qgate::QstateIdx;
using qgate::QstateSize;

using qgate::Qone;
using qgate::Qtwo;


template<class real>
CUDAQubitProcessor<real>::CUDAQubitProcessor(CUDADevices &devices) : devices_(devices) { }

template<class real>
CUDAQubitProcessor<real>::~CUDAQubitProcessor() {
    reset();
}

template<class real>
void CUDAQubitProcessor<real>::reset() {
    /* reset internal states */
    for (auto &proc : procs_)
        delete proc;
    procs_.clear();
    activeDevices_.clear();
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
initializeQubitStates(qgate::QubitStates &qstates, int nLanes) {
    CUQStates &cuQstates = static_cast<CUQStates&>(qstates);

    int po2idx = nLanes + (sizeof(DeviceComplex) / 8) + 2;
    MultiDeviceChunk *mchunk = cudaMemoryStore.allocate(po2idx);
    cuQstates.setMultiDeviceChunk(mchunk, nLanes);
    
    for (int idx = 0; idx < mchunk->getNChunks(); ++idx) {
        const DeviceChunk &chunk = mchunk->get(idx);
        procs_.push_back(new DeviceProcPrimitives<real>(*chunk.device));
        activeDevices_.push_back(chunk.device);
    }
    /* remove duplicates in activeDevices_. */
    auto lessDeviceNumber = [](const CUDADevice *dev0, const CUDADevice *dev1) {
        return dev0->getDeviceIdx() < dev1->getDeviceIdx();
    };
    std::sort(activeDevices_.begin(), activeDevices_.end(), lessDeviceNumber);
    auto eqDeviceNumber = [](const CUDADevice *dev0, const CUDADevice *dev1) {
        return dev0->getDeviceIdx() == dev1->getDeviceIdx();
    };
    auto duplicates = std::unique(activeDevices_.begin(), activeDevices_.end(), eqDeviceNumber);
    activeDevices_.erase(duplicates, activeDevices_.end());
}

template<class real>
void CUDAQubitProcessor<real>::resetQubitStates(qgate::QubitStates &qstates) {
    CUQStates &cuQstates = static_cast<CUQStates&>(qstates);
    DevicePtr &devPtr = cuQstates.getDevicePtr();
    auto resetFunc = [&](int procIdx, QstateIdx begin, QstateIdx end) {
        auto *proc = procs_[procIdx];
        proc->device().makeCurrent();
        proc->fillZero(devPtr, begin, end);
    };
    distribute(cuQstates, resetFunc);

    static const Complex cOne(1.);
    auto setOneFunc = [&](int procIdx, QstateIdx begin, QstateIdx end) {
        procs_[procIdx]->set(devPtr, &cOne, 0, sizeof(cOne));
    };
    distribute(cuQstates, setOneFunc, 0, 1);
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
    qgate::IdList relocated = qgate::relocateProcessors(cuQstates, localLane, -1);
    const DevicePtr &devPtr = cuQstates.getDevicePtr();
    auto calcProbLaunch = [&](int procIdx, QstateIdx begin, QstateIdx end) {
        int relocatedIdx = relocated[procIdx];
        auto *proc = procs_[relocatedIdx];
        proc->device().makeCurrent();
        proc->calcProb_launch(devPtr, localLane, begin, end);
    };

    QstateSize nStates = Qone << cuQstates.getNLanes();
    distribute((int)relocated.size(), calcProbLaunch, nStates / 2);
    
    std::vector<real> partialSum(relocated.size());
    auto calcProbSync = [&](int procIdx) {
        int relocatedIdx = relocated[procIdx];
        auto *proc = procs_[relocatedIdx];
        proc->device().makeCurrent();
        partialSum[procIdx] = proc->calcProb_sync();
    };
    qgate::Parallel((int)relocated.size()).run(calcProbSync);
    synchronize();

    real prob = std::accumulate(partialSum.begin(), partialSum.end(), real(0.));
    return prob;
}


template<class real>
void CUDAQubitProcessor<real>::join(qgate::QubitStates &_qstates,
                                    const QubitStatesList &qstatesList, int nNewLanes) {
    CUQStates &cuQstates = static_cast<CUQStates&>(_qstates);
    DevicePtr &dstPtr = cuQstates.getDevicePtr();
    QstateSize dstSize = Qone << cuQstates.getNLanes();
    int nSrcQstates = (int)qstatesList.size();
    
    assert(0 < nSrcQstates); 
    if (nSrcQstates == 1) {
        const CUDAQubitStates<real> *qsSrc =
                static_cast<const CUDAQubitStates<real>*>(qstatesList[0]);
        const DevicePtr &srcPtr = qsSrc->getDevicePtr();
        QstateSize srcSize = Qone << qsSrc->getNLanes();
        auto copyAndFillZeroFunc = [&](int chunkIdx, QstateIdx begin, QstateIdx end) {
            auto *proc = procs_[chunkIdx];
            proc->device().makeCurrent();
            proc->copyAndFillZero(dstPtr, srcPtr, srcSize, begin, end);
        };
        distribute(cuQstates, copyAndFillZeroFunc);
        synchronizeMultiDevice();
        return;
    }

    /* create list of buffer and size */
    std::valarray<const CUQStates*> srcQstatesList(nSrcQstates);
    std::valarray<QstateSize> srcSizeList(nSrcQstates);
    for (int idx = 0; idx < nSrcQstates; ++idx) {
        const CUQStates *qs = static_cast<const CUQStates*>(qstatesList[idx]);
        srcQstatesList[idx] = qs;
        srcSizeList[idx] = Qone << qs->getNLanes();
    }

    /* FIXME: procs should be relocated. */

    /* first kron, write to dst */
    int nSrcM2 = nSrcQstates - 2, nSrcM1 = nSrcQstates - 1;
    const DevicePtr &srcPtr0 = srcQstatesList[nSrcM2]->getDevicePtr(),
            &srcPtr1 = srcQstatesList[nSrcM1]->getDevicePtr();
    QstateSize Nsrc0 = srcSizeList[nSrcM2], Nsrc1 = srcSizeList[nSrcM1];
    QstateSize productSize = Nsrc0 * Nsrc1;

    /* FIXME: optimize */
    auto kronFunc = [&](int chunkIdx, QstateIdx begin, QstateIdx end) {
        procs_[chunkIdx]->kron(dstPtr, srcPtr0, Nsrc0, srcPtr1, Nsrc1, begin, end);
    };
    distribute(cuQstates, kronFunc, 0LL, productSize);
    synchronizeMultiDevice();
    
    for (int idx = nSrcQstates - 3; 0 <= idx; --idx) {
        const DevicePtr &srcPtr = srcQstatesList[idx]->getDevicePtr();
        QstateSize Nsrc = srcSizeList[idx];
        auto kronInPlaceFunc_0 = [&](int chunkIdx, QstateIdx begin, QstateIdx end) {
            procs_[chunkIdx]->kronInPlace_0(dstPtr, productSize, srcPtr, Nsrc, begin, end);
        };
        distribute(cuQstates, kronInPlaceFunc_0, productSize, productSize * Nsrc);
        synchronizeMultiDevice();
        
        auto kronInPlaceFunc_1 = [&](int chunkIdx, QstateIdx begin, QstateIdx end) {
            procs_[chunkIdx]->kronInPlace_1(dstPtr, productSize, srcPtr, Nsrc, begin, end);
        };
        distribute(cuQstates, kronInPlaceFunc_1, 0LL, productSize);
        synchronizeMultiDevice();
        
        productSize *= Nsrc;
    }

    /* zero fill */
    if (productSize != dstSize) {
        auto zeroFunc = [&](int chunkIdx, QstateIdx spanBegin, QstateIdx spanEnd) {
            auto *proc = procs_[chunkIdx];
            proc->device().makeCurrent();
            proc->fillZero(dstPtr, spanBegin, spanEnd);
        };
        distribute(cuQstates, zeroFunc, productSize, dstSize);
        synchronizeMultiDevice();
    }
}
    
template<class real>
void CUDAQubitProcessor<real>::decohere(int value, double prob,
                                        qgate::QubitStates &_qstates, int localLane) {

    CUQStates &cuQstates = static_cast<CUQStates&>(_qstates);
    DevicePtr &devPtr = cuQstates.getDevicePtr();

    /* decohere and shrink qubit states. */
    /* set bit */
    auto setBitFunc = [&](int chunkIdx, QstateIdx begin, QstateIdx end){
        auto *proc = procs_[chunkIdx];
        proc->device().makeCurrent();
        proc->decohere(devPtr, localLane, value, (real)prob, begin, end);
    };
    QstateSize nThreads = Qone << cuQstates.getNLanes();
    int nChunks = 1 << (cuQstates.getNLanes() - cuQstates.getNLanesPerChunk());
    distribute(nChunks, setBitFunc, nThreads);
    synchronizeMultiDevice();
}

template<class real> void CUDAQubitProcessor<real>::
decohereAndSeparate(int value, double prob,
                    qgate::QubitStates &_qstates0, qgate::QubitStates &_qstates1,
                    const qgate::QubitStates &_qstates, int localLane) {
    const CUDAQubitStates<real> &srcCuQstates = static_cast<const CUDAQubitStates<real>&>(_qstates);
    CUDAQubitStates<real> &dstCuQstates0 = static_cast<CUDAQubitStates<real>&>(_qstates0);
    CUDAQubitStates<real> &dstCuQstates1 = static_cast<CUDAQubitStates<real>&>(_qstates1);

    DevicePtr &dstDevPtr0 = dstCuQstates0.getDevicePtr();
    DevicePtr &dstDevPtr1 = dstCuQstates1.getDevicePtr();
    const DevicePtr &srcDevPtr = srcCuQstates.getDevicePtr();
    
    /* decohere and shrink qubit states. */
    qgate::IdList relocated;
    /* FIXME: mixture of src/dest procs are required. */
    if (value == 0)
        relocated = qgate::relocateProcessors(srcCuQstates, localLane, -1);
    else
        relocated = qgate::relocateProcessors(srcCuQstates, {localLane}, -1, -1);
    
    auto decohereFunc = [&](int chunkIdx, QstateIdx begin, QstateIdx end){
        int relocatedIdx = relocated[chunkIdx];
        auto *proc = procs_[relocatedIdx];
        proc->device().makeCurrent();
        proc->decohereAndShrink(dstDevPtr0, localLane, value,
                                (real)prob, srcDevPtr, begin, end);
    };
    QstateSize nThreads = Qone << dstCuQstates0.getNLanes();
    distribute((int)relocated.size(), decohereFunc, nThreads);

    static const Complex zero[2] = { 1., 0. };
    static const Complex one[2] = { 0., 1. };

    auto setFunc = [&](int chunkIdx, QstateIdx begin, QstateIdx end) {
        if (value == 0)
            procs_[chunkIdx]->set(dstDevPtr1, zero, 0, sizeof(DeviceComplex) * 2);
        else
            procs_[chunkIdx]->set(dstDevPtr1, one, 0, sizeof(DeviceComplex) * 2);
    };
    distribute(dstCuQstates1, setFunc, 0LL, 2LL);
    synchronizeMultiDevice();
}

template<class real>
void CUDAQubitProcessor<real>::applyReset(qgate::QubitStates &qstates, int localLane) {
    
    CUQStates &cuQstates = static_cast<CUQStates&>(qstates);
    DevicePtr &devPtr = cuQstates.getDevicePtr();

    qgate::IdList relocated = qgate::relocateProcessors(cuQstates, -1, localLane);
    auto resetFunc = [&](int chunkIdx, QstateIdx begin, QstateIdx end) {
        int relocatedIdx = relocated[chunkIdx];
        auto *proc = procs_[relocatedIdx];
        proc->device().makeCurrent();
        proc->applyReset(devPtr, localLane, begin, end);
    };
    distribute(cuQstates, resetFunc);
    synchronizeMultiDevice();
}


template<class real>
void CUDAQubitProcessor<real>::applyUnaryGate(const Matrix2x2C64 &mat, qgate::QubitStates &qstates, int localLane) {
    CUQStates &cuQstates = static_cast<CUQStates&>(qstates);
    DevicePtr &devPtr = cuQstates.getDevicePtr();

    DeviceMatrix2x2C<real> dmat(mat);

    qgate::IdList relocated = qgate::relocateProcessors(cuQstates, -1, localLane);
    auto applyUnaryGateFunc = [&](int procIdx, QstateIdx begin, QstateIdx end) {
        int relocatedIdx = relocated[procIdx];
        auto *proc = procs_[relocatedIdx];
        proc->device().makeCurrent();
        proc->applyUnaryGate(dmat, devPtr, localLane, begin, end);
    };
    QstateSize nThreads = Qone << (cuQstates.getNLanes() - 1);
    distribute((int)relocated.size(), applyUnaryGateFunc, nThreads);
    synchronizeMultiDevice();
}


template<class real> void CUDAQubitProcessor<real>::
applyControlGate(const Matrix2x2C64 &mat, QubitStates &qstates,
                 const qgate::IdList &localControlLanes, int localTargetLane) {
    CUQStates &cuQstates = static_cast<CUQStates&>(qstates);
    DevicePtr &devPtr = cuQstates.getDevicePtr();

    int nInputBits = (int)localControlLanes.size() + 1;
    int nIdxBits = qstates.getNLanes() - nInputBits;

    qgate::IdList allBitPos = localControlLanes;
    allBitPos.push_back(localTargetLane);
    qgate::IdList bitShiftMap = qgate::createBitShiftMap(allBitPos, nIdxBits);
    qgate::BitPermTable perm;
    perm.init_idxToQstateIdx(bitShiftMap);

    QstateIdx allControlBits = qgate::createBitmask(localControlLanes);
    QstateIdx targetBit = Qone << localTargetLane;
    
    std::vector<qgate::QstateIdxTable256*> d_tableList;
    for (int idx = 0; idx < (int)procs_.size(); ++idx) {
        DeviceProcPrimitives<real> *proc = procs_[idx];
        CUDADevice &device = proc->device();
        /* FIXME: remove SimpleMemoryStore for device. */
        SimpleMemoryStore &devMemStore = device.tempDeviceMemory();
        /* get 6 perm tables ( 48 bits ), FIXME: refactor */
        enum { nTables = 6 };
        auto *d_tables = devMemStore.allocate<qgate::QstateIdxTable256>(nTables);
        d_tableList.push_back(d_tables);
        device.makeCurrent();
        throwOnError(cudaMemcpyAsync(d_tables, perm.getTables(),
                                     sizeof(QstateIdx[256]) * nTables, cudaMemcpyDefault));
    }

    qgate::IdList relocated = qgate::relocateProcessors(cuQstates, localControlLanes, -1, localTargetLane);
    DeviceMatrix2x2C<real> dmat(mat);
    auto applyControlGateFunc = [&](int procIdx, QstateIdx begin, QstateIdx end) {
        int relocatedIdx = relocated[procIdx];
        auto *proc = procs_[relocatedIdx];
        proc->device().makeCurrent();
        proc->applyControlGate(dmat, devPtr, d_tableList[relocatedIdx],
			       allControlBits, targetBit, begin, end);
    };
    
    QstateSize nThreads = Qone << nIdxBits;
    distribute((int)relocated.size(), applyControlGateFunc, nThreads);
    synchronizeMultiDevice();
    /* temp device memory allocated in DeviceProcPrimitives. FIXME: allow delayed reset. */
    for (auto &device : activeDevices_)
        device->tempDeviceMemory().reset();  /* freeing device memory */
}

template<class real> template<class F> void CUDAQubitProcessor<real>::
distribute(int nProcs, const F &f, QstateSize nThreads) {
    QstateSize nThreadsPerProc = qgate::divru(nThreads, nProcs);
    for (int iProc = 0; iProc < nProcs; ++iProc) {
        QstateIdx beginInProc = std::min(nThreadsPerProc * iProc, nThreads);
        QstateIdx endInProc = std::min(nThreadsPerProc * (iProc + 1), nThreads);
        if (beginInProc != endInProc)
            f(iProc, beginInProc, endInProc);
    }
}


template<class real>template<class F>
void CUDAQubitProcessor<real>::distribute(const CUQStates &cuQstates, const F &f) {
    QstateIdx nThreads = Qone << cuQstates.getNLanes();
    distribute(cuQstates, f, 0, nThreads);
}

template<class real>template<class F>
void CUDAQubitProcessor<real>::distribute(const CUQStates &cuQstates, const F &f, QstateIdx begin, QstateIdx end) {
    QstateSize chunkSize = Qone << cuQstates.getNLanesPerChunk();
    int procBegin = int(begin / chunkSize);
    int procEnd = int((end + chunkSize - 1) / chunkSize);
    for (int procIdx = procBegin; procIdx < procEnd; ++procIdx) {
        QstateIdx spanBegin = std::max(procIdx * chunkSize, begin);
        QstateIdx spanEnd = std::min((procIdx + 1) * chunkSize, end);
        f(procIdx, spanBegin, spanEnd);
    }
}


template class CUDAQubitProcessor<float>;
template class CUDAQubitProcessor<double>;
