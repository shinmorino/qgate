#include "DeviceGetStates.h"
#include "DeviceFunctors.cuh"
#include "DeviceParallel.h"
#include <queue>

using namespace qgate_cuda;
using qgate::Qone;
using qgate::QstateIdx;
using qgate::QstateSize;


namespace {

template<class V> struct DeviceType;
template<> struct DeviceType<float> { typedef float Type; };
template<> struct DeviceType<double> { typedef double Type; };
template<> struct DeviceType<qgate::ComplexType<float>> { typedef DeviceComplexType<float> Type; };
template<> struct DeviceType<qgate::ComplexType<double>> { typedef DeviceComplexType<double> Type; };

}



template<class real>
DeviceGetStates<real>::DeviceGetStates(const qgate::QubitStatesList &qStatesList,
                                       CUDADeviceList &activeDevices) {

    activeDevices_ = activeDevices;
    int nQstates = (int)qStatesList.size();
    
    /* initialize list of qreg id list */
    IdList *idLists = new IdList[nQstates];
    DevicePtr *qStatesPtr = new DevicePtr[nQstates];
    memset(idLists, 0, sizeof(IdList) * nQstates);
    /* pack qreg id list */
    for (int qStatesIdx = 0; qStatesIdx < (int)nQstates; ++qStatesIdx) {
        const CUDAQubitStates<real> &cuQstates =
                static_cast<const CUDAQubitStates<real>&>(*qStatesList[qStatesIdx]);
        /* qregIds, qStatesPtr */
        const qgate::IdList &qregIds = cuQstates.getQregIdList();
        idLists[qStatesIdx].size = (int)qregIds.size();
        memcpy(idLists[qStatesIdx].id, qregIds.data(), sizeof(int) * qregIds.size());
        /* qstates ptr */
        qStatesPtr[qStatesIdx] = cuQstates.getDevicePtr();
    }
    /* create contexts */
    contexts_ = new GetStatesContext[nQstates];
    memset(contexts_, 0, sizeof(GetStatesContext) * nQstates);
    
    for (int idx = 0; idx < (int)activeDevices_.size(); ++idx) {
        GetStatesContext &ctx = contexts_[idx];
        ctx.device = activeDevices_[idx];
        ctx.device->makeCurrent();
        ctx.nQstates = nQstates;
        SimpleMemoryStore dmemStore = ctx.device->tempDeviceMemory();
        ctx.d_idLists = dmemStore.allocate<IdList>(ctx.nQstates);
        throwOnError(cudaMemcpyAsync(ctx.d_idLists, idLists,
                                     sizeof(IdList) * nQstates, cudaMemcpyDefault));
        ctx.d_qStatesPtr = dmemStore.allocate<DevicePtr>(ctx.nQstates);
        throwOnError(cudaMemcpyAsync(ctx.d_qStatesPtr, qStatesPtr,
                                     sizeof(DevicePtr) * nQstates, cudaMemcpyDefault));
    }
    for (int idx = 0; idx < (int)activeDevices_.size(); ++idx)
        contexts_[idx].device->synchronize();

    delete [] idLists;
    delete [] qStatesPtr;
}

template<class real>
DeviceGetStates<real>::~DeviceGetStates() {
    delete [] contexts_;
}


template<class real>
void DeviceGetStates<real>::run(void *array, qgate::QstateIdx arrayOffset, qgate::MathOp op,
                                qgate::QstateIdx begin, qgate::QstateIdx end) {
    
    switch (op) {
    case qgate::mathOpNull: {
        Complex *cmpArray = static_cast<Complex*>(array);
        run(&cmpArray[arrayOffset], null<real>(), begin, end);
        break;
    }
    case qgate::mathOpProb: {
        real *vArray = static_cast<real*>(array);
        run(&vArray[arrayOffset], abs2<real>(), begin, end);
        break;
    }
    default:
        abort_("Unknown math op.");
    }
}


template<class real> template<class R, class F>
void DeviceGetStates<real>::run(R *values, const F &op,
                                qgate::QstateIdx begin, qgate::QstateIdx end) {
    typedef typename DeviceType<R>::Type DeviceR;

    std::queue<GetStatesContext*> running;

    
    SimpleMemoryStore hMemStore = contexts_[0].device->tempHostMemory();
    stride_ = (int)hMemStore.capacity<DeviceR>();
    for (int idx = 0; idx < (int)activeDevices_.size(); ++idx) {
        GetStatesContext &ctx = contexts_[idx];
        ctx.device->makeCurrent();
        ctx.h_values = hMemStore.allocate<DeviceR>(stride_);
    }
    
    begin_ = begin;
    pos_ = begin;
    end_ = end;
    
    /* FIXME: pipeline */
    for (int idx = 0; idx < (int)activeDevices_.size(); ++idx) {
        if (!launch<R, F>(contexts_[idx], op))
            break;
        running.push(&contexts_[idx]);
    }
    
    while (!running.empty()) {
        GetStatesContext *ctx = running.front();
        running.pop();
        syncAndCopy(values, *ctx);
        if (launch<R, F>(*ctx, op))
            running.push(ctx);
    }
}

template<class real> template<class R, class F>
bool DeviceGetStates<real>::launch(GetStatesContext &ctx, const F &op) {

    if (pos_ == end_)
        return false;

    ctx.begin = pos_;
    ctx.end = std::min(pos_ + stride_, end_);
    
    GetStatesContext devCtx = ctx;
    QstateIdx offset = ctx.begin;
    
    typedef typename DeviceType<R>::Type DeviceR;
    auto calcStatesFunc = [=]__device__(QstateIdx globalIdx) {                 
        DeviceR v = DeviceR(1.);
        for (int iQstates = 0; iQstates < devCtx.nQstates; ++iQstates) {
            /* getStateByGlobalIdx() */
            const IdList &d_qregIds = devCtx.d_idLists[iQstates];
            QstateIdx localIdx = 0;
            for (int lane = 0; lane < d_qregIds.size; ++lane) {
                int qregId = d_qregIds.id[lane]; 
                if ((Qone << qregId) & globalIdx)
                    localIdx |= Qone << lane;
            }
            const DeviceComplex &state = devCtx.d_qStatesPtr[iQstates][localIdx];
            v *= op(state);
        }
        ((DeviceR*)devCtx.h_values)[globalIdx - offset] = v;
    };
    transform(ctx.begin, ctx.end, calcStatesFunc);
    pos_ = ctx.end;

    return true;
}

template<class real> template<class R>
void DeviceGetStates<real>::syncAndCopy(R *values, GetStatesContext &ctx) {
    ctx.device->synchronize();
    memcpy(&values[ctx.begin - begin_], ctx.h_values, sizeof(R) * (ctx.end - ctx.begin));
}

template class DeviceGetStates<float>;
template class DeviceGetStates<double>;

