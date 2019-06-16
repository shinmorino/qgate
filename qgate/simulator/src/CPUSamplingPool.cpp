#include "CPUSamplingPool.h"


using namespace qgate_cpu;
using qgate::QstateIdx;


template<class V>
CPUSamplingPool<V>::CPUSamplingPool(V *prob, int nLanes,
                                    const qgate::IdList &emptyLanes) {
    nStates_ = qgate::Qone << nLanes;
    V sum = V();
    for (QstateIdx idx = 0; idx < nStates_; ++idx) {
        sum += prob[idx];
        prob[idx] = sum;
    }
    V norm = V(1) / prob[nStates_ - 1];
    for (QstateIdx idx = 0; idx < nStates_; ++idx)
        prob[idx] *= norm;
    cumprob_ = prob;

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
    for (int idx = 0; idx < nSamples; ++idx) {
        V *v = std::upper_bound(cumprob_, cumprob_ + nStates_, (V)rnum[idx]);
        QstateIdx obs = v - cumprob_;
        observations[idx] = perm_.permute(obs);
    }
}


template class CPUSamplingPool<float>;
template class CPUSamplingPool<double>;
