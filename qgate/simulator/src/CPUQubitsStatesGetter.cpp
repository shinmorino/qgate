#include "CPUQubitsStatesGetter.h"
#include "Types.h"
#include "Parallel.h"
#include "BitPermTable.h"
#include "CPUQubitStates.h"
#include "CPUSamplingPool.h"

using qgate::QstateIdx;
using qgate::QstateSize;
using qgate::MathOp;
using qgate::Parallel;
using qgate::QubitStatesList;
using qgate::BitPermTable;
using qgate_cpu::CPUQubitStates;

using qgate::Qone;

namespace qcpu = qgate_cpu;
using namespace qgate_cpu;

namespace {

template<class R>
inline R abs2(const std::complex<R> &c) {
    return c.real() * c.real() + c.imag() * c.imag();
}

template<class R>
inline std::complex<R> null(const std::complex<R> &c) {
    return c;
}


template<class real, class R, class F>
void qubitsGetValues(R *values, const F &func,
                     const qgate::IdList *laneTransTables, QstateIdx emptyLaneMask,
                     const QubitStatesList &qstatesList,
                     QstateSize nStates, QstateIdx begin, QstateIdx step) {
    
    int nQubitStates = (int)qstatesList.size();
    /* array of CPUQubitStates */
    const CPUQubitStates<real> **qstates = new const CPUQubitStates<real>*[nQubitStates];
    for (int idx = 0; idx < nQubitStates; ++idx)
        qstates[idx] = static_cast<const CPUQubitStates<real>*>(qstatesList[idx]);
    /* array of BitPermTable. */
    BitPermTable *perm = new BitPermTable[nQubitStates];
    for (int idx = 0; idx < nQubitStates; ++idx)
        perm[idx].init_LaneTransform(laneTransTables[idx]);

    auto fgetstates = [=, &func](QstateIdx extDstIdx) {
        R v = R(1.);
        QstateIdx extSrcIdx = begin + extDstIdx * step;
        if ((extSrcIdx & emptyLaneMask) == 0) {
            for (int qStatesIdx = 0; qStatesIdx < nQubitStates; ++qStatesIdx) {
                QstateIdx localIdx = perm[qStatesIdx].permute(extSrcIdx);
                const ComplexType<real> &state = qstates[qStatesIdx]->operator[](localIdx);
                v *= func(state);
            }
            values[extDstIdx] = v;
        }
        else {
            values[extDstIdx] = 0.;
        }
    };

    auto fgetstates256 = [=, &func](QstateIdx idx, QstateIdx spanBegin, QstateIdx spanEnd) {
        /* spanBegin and spanEnd are multiples of 256. */
        QstateIdx extSrcIdx = begin + spanBegin * step;
        QstateIdx extSrcIdx_prefix = qgate::roundDown(extSrcIdx, 256);
        int extSrcIdx_8bits = (int)(extSrcIdx % 256);
        enum { loopStep = 256 };
        for (QstateIdx dstIdx256 = spanBegin; dstIdx256 < spanEnd; dstIdx256 += loopStep) {
            bool isEmpty[256];
            R *v = &values[dstIdx256];
            for (int idx = 0; idx < loopStep; ++idx) {
                QstateIdx extSrcIdxTmp = extSrcIdx + idx * step;
                isEmpty[idx] = (extSrcIdxTmp & emptyLaneMask) != 0;
                v[idx] = isEmpty[idx] ? R(0.) : R(1.);
            }

            for (int qstatesIdx = 0; qstatesIdx < nQubitStates; ++qstatesIdx) {
                QstateIdx extSrcIdx_prefix_tmp = extSrcIdx_prefix;
                int extSrcIdx_8bits_tmp = extSrcIdx_8bits;
                QstateIdx cached = perm[qstatesIdx].permute_56bits(extSrcIdx_prefix);
                
                for (int idx = 0; idx < loopStep; ++idx) {
                    if (!isEmpty[idx]) {
                        QstateIdx localIdx = perm[qstatesIdx].permute_8bits(cached, extSrcIdx_8bits_tmp);
                        const ComplexType<real> &state = qstates[qstatesIdx]->operator[](localIdx);
                        v[idx] *= func(state);
                    }
                        
                    extSrcIdx_8bits_tmp += (int)step;
                    if ((extSrcIdx_8bits_tmp & ~0xff) == 0)
                        continue;
                    /* went over 256 elm boundary. */
                    extSrcIdx_prefix_tmp += extSrcIdx_8bits_tmp & ~0xff;
                    extSrcIdx_8bits_tmp &= 0xff;
                    /* update cache */
                    cached = perm[qstatesIdx].permute_56bits(extSrcIdx_prefix_tmp);
                }                
            }
            extSrcIdx_8bits += (int)(step * loopStep);
            extSrcIdx_prefix += extSrcIdx_8bits & ~0xff;
            extSrcIdx_8bits &= 0xff;
        }
    };

    if (nStates < 256) {
        Parallel().for_each(fgetstates, 0, nStates);
    }
    else if (256 < step) {
        Parallel().for_each(fgetstates, 0, nStates);
    }
    else {
        QstateSize nStates256 = qgate::roundDown(nStates, 256);
        Parallel().distribute(fgetstates256, 0LL, nStates256, 256LL);
        if (nStates256 != nStates)
            Parallel().for_each(fgetstates, nStates256, nStates);
    }

    delete[] perm;
    delete[] qstates;
}

}

template<class real> void CPUQubitsStatesGetter<real>::
getStates(void *array, QstateIdx arrayOffset,
          MathOp op,
          const qgate::IdList *laneTransTables, QstateIdx emptyLaneMask,
          const QubitStatesList &qstatesList,
          QstateIdx nStates, QstateIdx begin, QstateIdx step) {
    
    const qgate::QubitStates *qstates = qstatesList[0];
    if (sizeof(real) == sizeof(float))
        abortIf(qstates->getPrec() != qgate::precFP32, "Wrong type");
    if (sizeof(real) == sizeof(double))
        abortIf(qstates->getPrec() != qgate::precFP64, "Wrong type");

    switch (op) {
    case qgate::mathOpNull: {
        ComplexType<real> *cmpArray = static_cast<ComplexType<real>*>(array);
        qubitsGetValues<real>(&cmpArray[arrayOffset], null<real>,
                              laneTransTables, emptyLaneMask, qstatesList, nStates, begin, step);
        break;
    }
    case qgate::mathOpProb: {
        real *vArray = static_cast<real*>(array);
        qubitsGetValues<real>(&vArray[arrayOffset], abs2<real>,
                              laneTransTables, emptyLaneMask, qstatesList, nStates, begin, step);
        break;
    }
    default:
        abort_("Unknown math op.");
    }
}

template<class real> void CPUQubitsStatesGetter<real>::
prepareProbArray(void *_prob,
                 const qgate::IdListList &laneTransformTables,
                 const QubitStatesList &qstatesList,
                 int nLanes, int nHiddenLanes) {
    real *prob = static_cast<real*>(_prob);

    int nQubitStates = (int)qstatesList.size();
    /* array of CPUQubitStates */
    typedef std::vector<const CPUQubitStates<real>*> QstatesPtrList;
    QstatesPtrList qstatesPtrs;
    for (int idx = 0; idx < nQubitStates; ++idx) {
        const CPUQubitStates<real> *cpuQstates =
                static_cast<const CPUQubitStates<real>*>(qstatesList[idx]);
        qstatesPtrs.push_back(cpuQstates);
    }
    /* array of BitPermTable. */
    qgate::BitPermTable *perm = new qgate::BitPermTable[nQubitStates];
    for (int idx = 0; idx < nQubitStates; ++idx)
        perm[idx].init_LaneTransform(laneTransformTables[idx]);

    /* Get prob reducing max 4 hidden lanes. */
    int nLanesToRemove = std::min(nHiddenLanes, 4);
    int nDstLanes = (nLanes + nHiddenLanes) - nLanesToRemove;
    QstateSize dstSize = Qone << nDstLanes;

    real *dstProb;
    if (nHiddenLanes == nLanesToRemove)
        dstProb = prob;
    else
        dstProb = static_cast<real*>(malloc(sizeof(real) * dstSize));

    QstateIdx nSrcsToSum = Qone << nLanesToRemove;
    auto reduceFromSrc = [=](QstateIdx dstIdx) {
        QstateIdx srcBegin = dstIdx * nSrcsToSum;
        real sum = real();
        for (QstateIdx idx = srcBegin; idx < srcBegin + nSrcsToSum; ++idx) {
            real v = real(1.);
            for (int qStatesIdx = 0; qStatesIdx < nQubitStates; ++qStatesIdx) {
                QstateIdx localIdx = perm[qStatesIdx].permute(idx);
                const ComplexType<real> &state = qstatesPtrs[qStatesIdx]->operator[](localIdx);
                v *= abs2(state);
            }
            sum += v;
        }
        dstProb[dstIdx] = sum;
    };
    Parallel().for_each(reduceFromSrc, 0, dstSize);
    delete[] perm;

    if (nHiddenLanes == nLanesToRemove)
        return;

    nHiddenLanes -= nLanesToRemove;

    /* updated params for memory allocation */
    nLanesToRemove = std::min(nHiddenLanes, 4);
    nDstLanes = (nLanes + nHiddenLanes) - nLanesToRemove;
    dstSize = Qone << nDstLanes;
    /* manage src and dst buffer */
    real *srcProb = dstProb;
    if (nHiddenLanes == nLanesToRemove)
        dstProb = prob;
    else
        dstProb = static_cast<real*>(malloc(sizeof(real) * dstSize));

    while (true) {
        /* size parameters again for the next loop */
        nLanesToRemove = std::min(nHiddenLanes, 4);
        nDstLanes = (nLanes + nHiddenLanes) - nLanesToRemove;
        dstSize = Qone << nDstLanes;

        QstateIdx nSrcsToSum = Qone << nLanesToRemove;
        auto reduceProb = [=](QstateIdx dstIdx) {
            QstateIdx srcBegin = dstIdx * nSrcsToSum;
            real v = real(0.);
            for (QstateIdx idx = srcBegin; idx < srcBegin + nSrcsToSum; ++idx)
                v += srcProb[idx];
            dstProb[dstIdx] = v;
        };
        Parallel().for_each(reduceProb, 0, dstSize);

        nHiddenLanes -= nLanesToRemove;
        if (nHiddenLanes != 0) {
            std::swap(srcProb, dstProb);
            if (nHiddenLanes <= 4) {
                free(dstProb);
                dstProb = prob;
            }
        }
        else {
            break;
        }
    };
    free(srcProb);
}

template<class real> qgate::SamplingPool *CPUQubitsStatesGetter<real>::
createSamplingPool(const qgate::IdListList &laneTransformTables,
                   const QubitStatesList &qstatesList,
                   int nLanes, int nHiddenLanes, const qgate::IdList &emptyLanes) {
    QstateSize nStates = Qone << nLanes;
    real *prob = (real*)malloc(sizeof(real) * nStates);
    prepareProbArray(prob, laneTransformTables, qstatesList, nLanes, nHiddenLanes);
    return new CPUSamplingPool<real>(prob, nLanes, emptyLanes);
}

template class CPUQubitsStatesGetter<float>;
template class CPUQubitsStatesGetter<double>;
