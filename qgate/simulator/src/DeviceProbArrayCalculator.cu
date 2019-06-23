#include "DeviceProbArrayCalculator.h"
#include "DeviceParallel.h"
#include "TransferringRunner.h"
#include "CUDAGlobals.h"
#include "DeviceFunctors.cuh"

using namespace qgate_cuda;
using qgate::QstateIdx;
using qgate::QstateSize;
using qgate::Qone;

namespace {

template<class real, class Ptr>
void calcAndReduceProb(QstateIdx begin, QstateIdx end,
                       Ptr d_out, const DeviceQubitStates<real> *d_qsList, int nQstates,
                       int nDstLanes, int nLanesToReduce) {

    /* copies for capture. */
    QstateSize stride = Qone << nDstLanes;
    QstateIdx nSrcProbs = Qone << (nDstLanes + nLanesToReduce);

    auto reduceProb = [=]__device__(QstateIdx extIdx) mutable {
        real reduced = real();
        for (QstateIdx extSrcIdx = extIdx; extSrcIdx < nSrcProbs; extSrcIdx += stride) {
            real v = real(1.);
            for (int iQstates = 0; iQstates < nQstates; ++iQstates) {
                /* getStateByGlobalIdx() */
                const DeviceQubitStates<real> &d_qs = d_qsList[iQstates];
                QstateIdx localSrcIdx = 0;
                for (int iTable = 0; iTable < DeviceQubitStates<real>::nTables; ++iTable) {
                    int iByte = (extSrcIdx >> (8 * iTable)) & 0xff;
                    localSrcIdx |= d_qs.bitPermTable[iTable][iByte];
                }
                const DeviceComplexType<real> &state = d_qs.ptr[localSrcIdx];
                v *= abs2<real>()(state);
            }
            reduced += v;
        }
        d_out[extIdx] = reduced;
    };
    transform(begin, end, reduceProb);
}

}

template<class real>
struct CalcAndReduceProbWorker : public DeviceWorker {

    CalcAndReduceProbWorker(const DeviceQubitStates<real> *d_qsList, int nQstates,
                            int nLanes, int nLanesToReduce, CUDADevice *device) : DeviceWorker(device) {
        d_qsList_ = d_qsList;
        nQstates_ = nQstates;
        nLanes_ = nLanes;
        nLanesToReduce_ = nLanesToReduce;
    }
    
    virtual void run(void *_dst, QstateIdx begin, QstateIdx end) {
        real *dst = static_cast<real*>(_dst);
        calcAndReduceProb<real, real*>(begin, end, dst, d_qsList_, nQstates_, nLanes_, nLanesToReduce_);
    }
    
    const DeviceQubitStates<real> *d_qsList_;
    int nQstates_;
    int nLanes_, nLanesToReduce_;
};


template<class real, class PtrOut>
void reduceProb(QstateIdx begin, QstateIdx end, PtrOut &d_out,
                const MultiChunkPtr<real> &d_in, int nLanes, int nLanesToReduce) {

    QstateSize stride = Qone << nLanes;
    QstateSize nInputs = Qone << (nLanes + nLanesToReduce);

    auto reduceProb = [=] __device__ (QstateIdx extIdx) mutable {
        real reduced = real();
        for (QstateIdx srcIdx = extIdx; srcIdx < nInputs; srcIdx += stride)
            reduced += d_in[srcIdx];
        d_out[extIdx] = reduced;
    };

    transform(begin, end, reduceProb);
}


template<class real>
struct ReduceProbWorker : public DeviceWorker {
    
    ReduceProbWorker(const MultiChunkPtr<real> &d_probIn, int nLanes, int nLanesToReduce,
                     CUDADevice *device) : DeviceWorker(device) {
        d_probIn_ = d_probIn;
        nLanes_ = nLanes;
        nLanesToReduce_ = nLanesToReduce;
    }

    virtual void run(void *_dst, qgate::QstateIdx begin, qgate::QstateIdx end) {
        real *dst = static_cast<real*>(_dst);
        reduceProb(begin, end, dst, d_probIn_, nLanes_, nLanesToReduce_);
    }

    MultiChunkPtr<real> d_probIn_;
    int nLanes_, nLanesToReduce_;
};


template<class real> void DeviceProbArrayCalculator<real>::
setUp(const qgate::IdListList &laneTransformTables, const qgate::QubitStatesList &qsList,
      CUDADeviceList &devices) {

    devices_ = &devices;
    
    /* create contexts */
    nQstates_ = (int)qsList.size();
    std::vector<DeviceQstates> devQs(nQstates_);
    for (int qsIdx = 0; qsIdx < (int)devQs.size(); ++qsIdx) {
        /* CUDAQubitStates */
        const CUDAQubitStates<real> *cuQstates =
                static_cast<const CUDAQubitStates<real>*>(qsList[qsIdx]);
        /* BitPermTable */
        qgate::BitPermTable perm;
        perm.init_LaneTransform(laneTransformTables[qsIdx]);
        /* initialize DeviceInput. */
        devQs[qsIdx].set(cuQstates->getDevicePtr(), perm);
    }

    int nDevices = (int)devices.size();
    d_qsPtrs_.resize(nDevices);
    /* transferring DeviceQstates array to each device. */
    for (int iDevice = 0; iDevice < nDevices; ++iDevice) {
        CUDADevice *device = devices[iDevice];
        device->makeCurrent();
        /* FIXME: remove SimpleMemoryStore for device. */
        SimpleMemoryStore &dmemStore = device->tempDeviceMemory();
        auto *d_qs = dmemStore.allocate<DeviceQstates>(nQstates_);
        throwOnError(cudaMemcpyAsync(d_qs, devQs.data(),
                                     sizeof(DeviceQstates) * nQstates_, cudaMemcpyDefault));
        d_qsPtrs_[iDevice] = d_qs;
    }
    for (int idx = 0; idx < (int)devices.size(); ++idx)
        devices[idx]->synchronize();
}

template<class real>
void DeviceProbArrayCalculator<real>::tearDown() {
    for (auto *device : *devices_) {
        /* FIXME: refactor SimpleMemoryStore. */
        device->tempHostMemory().reset();  /* freeing host memory */
        device->tempDeviceMemory().reset();  /* freeing device memory */
    }
}

template<class real>
void DeviceProbArrayCalculator<real>::run(real *array, int nLanes, int nHiddenLanes) {
    if (nHiddenLanes <= 2)  {
        /* direct */
        runOneStepReduction(array, nLanes, nHiddenLanes);
    }
    else {
        /* multi-step */
        runMultiStepReduction(array, nLanes, nHiddenLanes);
    }
}

template<class real>
void DeviceProbArrayCalculator<real>::runOneStepReduction(real *array, int nLanes, int nHiddenLanes) {
    assert(nHiddenLanes <= 2);
    DeviceWorkers devWorkers;
    for (int idev = 0; idev < (int)devices_->size(); ++idev) {
        CUDADevice *device = (*devices_)[idev];
        auto *devWorker =
                new CalcAndReduceProbWorker<real>(d_qsPtrs_[idev], nQstates_,
                                                  nLanes, nHiddenLanes, device);
        devWorkers.push_back(devWorker);
    }

    QstateSize nStates = Qone << nLanes;
    run_d2h(array, devWorkers, 0, nStates);

    for (auto devWorker : devWorkers)
        delete devWorker;
}


template<class real>
void DeviceProbArrayCalculator<real>::runMultiStepReduction(real *array, int nLanes, int nHiddenLanes) {
    assert(nMaxLanesToReduce <= nHiddenLanes);
    const int valueSizeInPo2 = sizeof(real) / sizeof(float) + 1;

    /* calculate and reduce prob array */
    int nLanesToReduce = std::min(nHiddenLanes, (int)nMaxLanesToReduce);
    nHiddenLanes -= nLanesToReduce;
    int nDstLanes = nLanes + nHiddenLanes;
    MultiDeviceChunk *mDstChunk = cudaMemoryStore.allocate(nDstLanes + valueSizeInPo2);

    int nChunks = mDstChunk->getNChunks();
    QstateSize span = (Qone << nDstLanes) / nChunks;
    for (int idx = 0; idx < nChunks; ++idx) {
        CUDADevice *device = mDstChunk->get(idx).device;
        device->makeCurrent();
        QstateIdx begin = span * idx, end = span * (idx + 1);
        MultiChunkPtr<real> d_dst = mDstChunk->getMultiChunkPtr<real>();
        calcAndReduceProb<real, MultiChunkPtr<real>>(begin, end, d_dst, d_qsPtrs_[idx], nQstates_,
                                                     nDstLanes, nLanesToReduce);
    }
    if (nChunks != 1) {
        for (int idev = 0; idev < nChunks; ++idev) {
            CUDADevice *device = mDstChunk->get(idev).device;
            device->synchronize();
        }
    }

    /* reduce prob array */
    /* update size */
    nLanesToReduce = std::min(nHiddenLanes, (int)nMaxLanesToReduce);
    nHiddenLanes -= nLanesToReduce; /* represents nHiddenLanes after reduction */
    nDstLanes = nLanes + nHiddenLanes;

    MultiDeviceChunk *mSrcChunk = mDstChunk;
    mDstChunk = NULL;
    if (nHiddenLanes != 0) {
        /* reduce on device */
        mDstChunk = cudaMemoryStore.allocate(nDstLanes + valueSizeInPo2);

        while (nHiddenLanes != 0) {
            MultiChunkPtr<real> d_dst = mDstChunk->getMultiChunkPtr<real>();
            MultiChunkPtr<real> d_src = mSrcChunk->getMultiChunkPtr<real>();
            /* FIXME: introduce mininum span */
            QstateSize span = (Qone << nDstLanes) / nChunks;
            for (int idx = 0; idx < mDstChunk->getNChunks(); ++idx) {
                CUDADevice *device = mDstChunk->get(idx).device;
                device->makeCurrent();
                QstateIdx begin = span * idx, end = span * (idx + 1);
                reduceProb(begin, end, d_dst, d_src, nDstLanes, nLanesToReduce);
            }
            if (nChunks != 1) {
                for (int idev = 0; idev < nChunks; ++idev) {
                    CUDADevice *device = mDstChunk->get(idev).device;
                    device->synchronize();
                }
            }
            std::swap(mSrcChunk, mDstChunk);
            nLanesToReduce = std::min(nHiddenLanes, (int)nMaxLanesToReduce);
            nHiddenLanes -= nLanesToReduce; /* the value after reduction */
            nDstLanes = nLanes + nHiddenLanes;
        }
        cudaMemoryStore.deallocate(mDstChunk);
    }
    
    /* reduce prob array and transfer to host */
    DeviceWorkers devWorkers;
    MultiChunkPtr<real> d_in = mSrcChunk->getMultiChunkPtr<real>();
    for (int idev = 0; idev < (int)devices_->size(); ++idev) {
        CUDADevice *device = (*devices_)[idev];
        auto *devWorker =
                new ReduceProbWorker<real>(d_in, nLanes, nLanesToReduce, device);
        devWorkers.push_back(devWorker);
    }

    QstateSize nStates = Qone << nLanes;
    run_d2h(array, devWorkers, 0, nStates);
    
    for (auto *devWorker : devWorkers)
        delete devWorker;
    cudaMemoryStore.deallocate(mSrcChunk);
}

template class qgate_cuda::DeviceProbArrayCalculator<float>;
template class qgate_cuda::DeviceProbArrayCalculator<double>;
