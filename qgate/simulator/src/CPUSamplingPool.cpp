#include "CPUSamplingPool.h"
#include "Parallel.h"

using namespace qgate_cpu;
using qgate::QstateIdx;


template<class V>
CPUSamplingPool<V>::CPUSamplingPool(V *prob, int nLanes,
                                    const qgate::IdList &emptyLanes) {
    nStates_ = qgate::Qone << nLanes;

    qgate::Parallel parallel;
    int nWorkers = parallel.getNWorkers(nStates_);
    V *partialSum = new V[nWorkers]();
    auto prefixSumFunc = [=](int threadIdx, QstateIdx spanBegin, QstateIdx spanEnd) {
        V v = V(0.);
        for (QstateIdx idx = spanBegin; idx < spanEnd; ++idx) {
            v += prob[idx];
            prob[idx] = v;
        }
        partialSum[threadIdx] = v;
    };
    parallel.distribute(0, nStates_, prefixSumFunc);
    /* prefix sum for partial sum */
    V v = V();
    for (QstateIdx idx = 0; idx < nWorkers; ++idx) {
        v += partialSum[idx];
        partialSum[idx] = v;
    }
    /* add partial prefix sum and normalize */
    V norm = V(1) / partialSum[nWorkers - 1];
    auto applyPartialSum = [=](int threadIdx, QstateIdx spanBegin, QstateIdx spanEnd) {
        if (threadIdx != 0) {
            for (QstateIdx idx = spanBegin; idx < spanEnd; ++idx) {
                prob[idx] += partialSum[threadIdx - 1];
                prob[idx] *= norm;
            }
        }
    };
    parallel.distribute(0, nStates_, applyPartialSum);
    /* free partialSum array. */
    delete[] partialSum;

    /* set cumprob array */
    cumprob_ = prob;
#if 0
    for (QstateIdx idx = 0; idx < nStates_ - 1; ++idx) {
        V diff = cumprob_[idx + 1] - cumprob_[idx];
        abortIf(diff < 0.);
        abortIf(1. < diff);
    }
#endif
    /* prep permutation */
    qgate::IdList transform = qgate::createBitShiftMap(emptyLanes, nLanes);
    perm_.init_idxToQstateIdx(transform);
}

template<class V>
CPUSamplingPool<V>::~CPUSamplingPool() {
    free(cumprob_);
    cumprob_ = NULL;
}

template<class V>
void CPUSamplingPool<V>::sample(QstateIdx *observations, int nSamples, const double *rnum) {
    auto sampleFunc = [=](QstateIdx idx) {
        V *v = std::upper_bound(cumprob_, cumprob_ + nStates_, (V)rnum[idx]);
        QstateIdx obs = v - cumprob_;
        assert(0 <= obs);
        assert(obs < nStates_);
        observations[idx] = perm_.permute(obs);
    };
    qgate::Parallel().for_each(0, nSamples, sampleFunc);
}


template class CPUSamplingPool<float>;
template class CPUSamplingPool<double>;
