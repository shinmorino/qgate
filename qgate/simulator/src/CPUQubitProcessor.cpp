#include "CPUQubitProcessor.h"
#include "CPUQubitStates.h"
#include <algorithm>
#include <string.h>
#include "BitPermTable.h"
#include "Parallel.h"
#include <valarray>
#include "CPUSamplingPool.h"

using namespace qgate_cpu;
using qgate::Qone;
using qgate::Qtwo;
using qgate::Parallel;

template<class real>
CPUQubitProcessor<real>::CPUQubitProcessor() { }
template<class real>
CPUQubitProcessor<real>::~CPUQubitProcessor() { }

template<class real>
void CPUQubitProcessor<real>::reset() {
}

template<class real> void CPUQubitProcessor<real>::
initializeQubitStates(qgate::QubitStates &_qstates, int nLanes) {
    CPUQubitStates<real> &qstates = static_cast<CPUQubitStates<real>&>(_qstates);
    qstates.allocate(nLanes);
}

template<class real> template<class G>
void CPUQubitProcessor<real>::run(int nLanes,
                                  int nInputBits, const qgate::IdList &bitShiftMap, const G &gatef) {

    int nIdxBits = nLanes - nInputBits;
    qgate::BitPermTable perm;
    perm.init_idxToQstateIdx(bitShiftMap);

    QstateIdx nLoops = Qone << nIdxBits;
    if (nLoops < 256) {
        for (int idx = 0; idx < nLoops; ++idx) {
            QstateIdx idx_base = perm.permute_8bits(0, idx);
            gatef(idx_base, idx);
        }
    }
    else {
        auto gateFunc256 =
            [=, &perm](int threadIdx, QstateIdx spanBegin, QstateIdx spanEnd) {
            for (QstateIdx idx256 = spanBegin; idx256 < spanEnd; idx256 += 256) {
                QstateIdx idx56bits = perm.permute_56bits(idx256);
                for (int idx = 0; idx < 256; ++idx) {
                    QstateIdx idx_base = perm.permute_8bits(idx56bits, idx);
                    gatef(idx_base, idx256 + idx);
                }
            }
        };
        qgate::Parallel().distribute(gateFunc256, 0LL, nLoops, 256LL);
    }
}


template<class real>
void CPUQubitProcessor<real>::resetQubitStates(qgate::QubitStates &_qstates) {
    CPUQubitStates<real> &qstates = static_cast<CPUQubitStates<real>&>(_qstates);
    
    Complex *cmp = qstates.getPtr();
    qgate::QstateSize nStates = Qone << qstates.getNLanes();

    auto setZeroFunc = [=](int threadIdx, QstateIdx spanBegin, QstateIdx spanEnd) {
        memset(&cmp[spanBegin], 0, sizeof(Complex) * (spanEnd - spanBegin));
    };
    Parallel().distribute(setZeroFunc, 0LL, nStates);
    cmp[0] = Complex(1.);
}

template<class real> double CPUQubitProcessor<real>::
calcProbability(const qgate::QubitStates &_qstates, int localLane) {
    const CPUQubitStates<real> &qstates = static_cast<const CPUQubitStates<real>&>(_qstates);
    return _calcProbability(qstates, localLane);
}

template<class real>
real CPUQubitProcessor<real>::_calcProbability(const CPUQubitStates<real> &qstates, int localLane) {
    
    auto sumFunc = [=, &qstates](QstateIdx idx_0) {
        const Complex &qs = qstates[idx_0];
        return qs.real() * qs.real() + qs.imag() * qs.imag();
    };

    int nLanes = qstates.getNLanes();
    int nIdxBits = nLanes - 1;
    qgate::IdList bitShiftMap = qgate::createBitShiftMap(localLane, nIdxBits);
    qgate::BitPermTable perm;
    perm.init_idxToQstateIdx(bitShiftMap);
    
    QstateIdx nLoops = Qone << nIdxBits;
    real prob = real(0.);
    if (nLoops < 256) {
        for (int idx = 0; idx < nLoops; ++idx) {
            prob += sumFunc(perm.permute_8bits(0, idx));
        }
    }
    else {
        qgate::Parallel parallel;
        int nWorkers = parallel.getNWorkers(nLoops);
        real *partialSum = new real[nWorkers]();
        auto forloop = [=](int threadIdx, QstateIdx spanBegin, QstateIdx spanEnd) {
                           real v = real(0.);
                           for (QstateIdx idx256 = spanBegin; idx256 < spanEnd; idx256 += 256) {
                               QstateIdx idx56bits = perm.permute_56bits(idx256);
                               for (int idx = 0; idx < 256; ++idx) {
                                   QstateIdx idx_base = perm.permute_8bits(idx56bits, idx);
                                   v += sumFunc(idx_base);
                               }
                           }
                           partialSum[threadIdx] = v;
                       };
        parallel.distribute(forloop, 0LL, nLoops, 256LL);
        prob = real(0.);
        for (int idx = 0; idx < nWorkers; ++idx) {
            prob += partialSum[idx];
        }
        delete[] partialSum;
    }
    return prob;
}



template<class real>
void CPUQubitProcessor<real>::join(qgate::QubitStates &_qstates,
                                   const QubitStatesList &qstatesList, int nNewLanes) {
    CPUQubitStates<real> &qstates = static_cast<CPUQubitStates<real>&>(_qstates);
    int nSrcQstates = (int)qstatesList.size();
    Complex *dst = qstates.getPtr();
    QstateSize dstSize = Qone << qstates.getNLanes();
    
    assert(0 < nSrcQstates); 
    if (nSrcQstates == 1) {
        const CPUQubitStates<real> *qs = static_cast<const CPUQubitStates<real>*>(qstatesList[0]);
        const Complex *src = qs->getPtr();
        QstateSize srcSize = Qone << qs->getNLanes();
        auto copyAndZeroFunc = [=](int threadIdx, QstateIdx spanBegin, QstateIdx spanEnd) {
            if (spanBegin < srcSize) {
                QstateIdx copyEnd = std::min(srcSize, spanEnd);
                QstateSize copySize = copyEnd - spanBegin;
                memcpy(&dst[spanBegin], &src[spanBegin], sizeof(Complex) * copySize);
            }
            if (srcSize < spanEnd) {
                QstateIdx zeroBegin = std::max(srcSize, spanBegin);
                QstateSize zeroSize = spanEnd - zeroBegin;
                memset(&dst[zeroBegin], 0, sizeof(Complex) * zeroSize);
            }
        };
        qgate::Parallel().distribute(copyAndZeroFunc, 0LL, dstSize);
        return;
    }

    /* create list of buffer and size */
    std::valarray<const Complex*> srcBufList(nSrcQstates);
    std::valarray<QstateSize> srcSizeList(nSrcQstates);
    for (int idx = 0; idx < nSrcQstates; ++idx) {
        const CPUQubitStates<real> *qs = static_cast<const CPUQubitStates<real>*>(qstatesList[idx]);
        srcBufList[idx] = qs->getPtr();
        srcSizeList[idx] = Qone << qs->getNLanes();
    }

    /* first kron, write to dst */
    int nSrcM2 = nSrcQstates - 2, nSrcM1 = nSrcQstates - 1;
    const Complex *src0 = srcBufList[nSrcM2], *src1 = srcBufList[nSrcM1];
    QstateSize Nsrc0 = srcSizeList[nSrcM2], Nsrc1 = srcSizeList[nSrcM1];
    QstateSize productSize = Nsrc0 * Nsrc1;
    auto kronFunc = [=](int threadIdx, QstateIdx spanBegin, QstateIdx spanEnd) {
        for (QstateIdx idx0 = 0; idx0 < Nsrc0; ++idx0) {
            Complex *dstIdx0 = &dst[idx0 * Nsrc1];
            Complex v0 = src0[idx0];
            /* parallelize this loop, N0 < N1 */
            for (QstateIdx idx1 = spanBegin; idx1 < spanEnd; ++idx1)
                dstIdx0[idx1] = v0 * src1[idx1];
        }
    };
    qgate::Parallel().distribute(kronFunc, 0LL, Nsrc1);
    
    for (int idx = nSrcQstates - 3; 0 <= idx; --idx) {
        const Complex *src = srcBufList[idx];
        QstateSize Nsrc = srcSizeList[idx];
        
        auto kronInPlaceF_0 = [=](int threadIdx, QstateIdx spanBegin, QstateIdx spanEnd) {
            for (QstateIdx idxSrc = 1; idxSrc < Nsrc; ++idxSrc) {
                Complex vSrc = src[idxSrc];
                Complex *dstTmp = &dst[idxSrc * productSize];
                for (QstateIdx idx = spanBegin; idx < spanEnd; ++idx)
                    dstTmp[idx] = vSrc * dst[idx];
            }
        };
        qgate::Parallel().distribute(kronInPlaceF_0, 0LL, productSize);
        
        Complex vSrc = src[0];
        auto kronInPlaceF_1 = [=](int threadIdx, QstateIdx spanBegin, QstateIdx spanEnd) {
            for (QstateIdx idx = spanBegin; idx < spanEnd; ++idx)
                dst[idx] *= vSrc;
        };
        qgate::Parallel().distribute(kronInPlaceF_1, 0LL, productSize);

        productSize *= Nsrc;
    }

    /* zero fill */
    auto zero = [=](int threadIdx, QstateIdx spanBegin, QstateIdx spanEnd) {
        QstateSize zeroSize = spanEnd - spanBegin;
        memset(&dst[spanBegin], 0, sizeof(Complex) * zeroSize);
    };
    qgate::Parallel().distribute(zero, productSize, dstSize);
}



template<class real>
void CPUQubitProcessor<real>::decohere(int value, double prob,
                                       qgate::QubitStates &_qstates, int localLane) {
    CPUQubitStates<real> &qstates = static_cast<CPUQubitStates<real>&>(_qstates);
    
    QstateIdx laneBit = Qone << localLane;
    
    std::function<void(QstateIdx, QstateIdx)> setBitFunc;
    
    if (value == 0) {
        real norm = real(1. / std::sqrt(prob));
        setBitFunc = [=, &qstates](QstateIdx idx_0, QstateIdx) {
            QstateIdx idx_1 = idx_0 | laneBit;
            qstates[idx_0] *= norm;
            qstates[idx_1] = real(0.);
        };
    }
    else {
        real norm = real(1. / std::sqrt(1. - prob));
        setBitFunc = [=, &qstates](QstateIdx idx_0, QstateIdx) {
            QstateIdx idx_1 = idx_0 | laneBit;
            qstates[idx_0] = real(0.);
            qstates[idx_1] *= norm;
        };
    }
    
    int nIdxBits = qstates.getNLanes() - 1;
    qgate::IdList bitShiftMap = qgate::createBitShiftMap(localLane, nIdxBits);
    run(qstates.getNLanes(), 1, bitShiftMap, setBitFunc);
}

template<class real> void CPUQubitProcessor<real>::
decohereAndSeparate(int value, double prob,
                    qgate::QubitStates &_qstates0, qgate::QubitStates &_qstates1,
                    const qgate::QubitStates &_qstates, int localLane) {
    const CPUQubitStates<real> &qstates = static_cast<const CPUQubitStates<real>&>(_qstates);
    CPUQubitStates<real> &qstates0 = static_cast<CPUQubitStates<real>&>(_qstates0);
    CPUQubitStates<real> &qstates1 = static_cast<CPUQubitStates<real>&>(_qstates1);
    
    QstateIdx laneBit = Qone << localLane;
    
    std::function<void(QstateIdx, QstateIdx)> decohereFunc;
    
    if (value == 0) {
        real norm = real(1. / std::sqrt(prob));
        decohereFunc = [=, &qstates, &qstates0](QstateIdx idx_0, QstateIdx idx) {
            qstates0[idx] = norm * qstates[idx_0];
        };
        /* set |0> */
        qstates1[0] = 1.;
        qstates1[1] = 0.;
    }
    else {
        real norm = real(1. / std::sqrt(1. - prob));
        decohereFunc = [=, &qstates, &qstates0](QstateIdx idx_0, QstateIdx idx) {
            QstateIdx idx_1 = idx_0 | laneBit;
            qstates0[idx] = norm * qstates[idx_1];
        };
        /* set |1> */
        qstates1[0] = 0.;
        qstates1[1] = 1.;
    }
    
    int nIdxBits = qstates.getNLanes() - 1;
    qgate::IdList bitShiftMap = qgate::createBitShiftMap(localLane, nIdxBits);
    /* nInputBits == 0, qstates0 does not have any input bits. */
    run(qstates0.getNLanes(), 0, bitShiftMap, decohereFunc);
}

template<class real>
void CPUQubitProcessor<real>::applyReset(qgate::QubitStates &_qstates, int localLane) {

    CPUQubitStates<real> &qstates = static_cast<CPUQubitStates<real>&>(_qstates);
    
    QstateIdx laneBit = Qone << localLane;
    
    /* Assuming reset is able to be applyed after measurement.
     * Ref: https://quantumcomputing.stackexchange.com/questions/3908/possibility-of-a-reset-quantum-gate */

    auto resetFunc = [=, &qstates](QstateIdx idx_0, QstateIdx) {
                         QstateIdx idx_1 = idx_0 | laneBit;
                         qstates[idx_0] = qstates[idx_1];
                         qstates[idx_1] = real(0.);
                     };
    
    int nIdxBits = qstates.getNLanes() - 1;
    qgate::IdList bitShiftMap = qgate::createBitShiftMap(localLane, nIdxBits);
    run(qstates.getNLanes(), 1, bitShiftMap, resetFunc);
}

template<class real>
void CPUQubitProcessor<real>::applyUnaryGate(const Matrix2x2C64 &_mat,
                                             qgate::QubitStates &_qstates, int localLane) {
    
    CPUQubitStates<real> &qstates = static_cast<CPUQubitStates<real>&>(_qstates);
    Matrix2x2CR mat(_mat);
    
    QstateIdx laneBit = Qone << localLane;

    auto unaryGateFunc = [=, &qstates](QstateIdx idx_0, QstateIdx) {
                             QstateIdx idx_1 = idx_0 | laneBit;
                             const Complex &qs0 = qstates[idx_0];
                             const Complex &qs1 = qstates[idx_1];
                             Complex qsout0 = mat(0, 0) * qs0 + mat(0, 1) * qs1;
                             Complex qsout1 = mat(1, 0) * qs0 + mat(1, 1) * qs1;
                             qstates[idx_0] = qsout0;
                             qstates[idx_1] = qsout1;
                         };
    
    int nIdxBits = qstates.getNLanes() - 1;
    qgate::IdList bitShiftMap = qgate::createBitShiftMap(localLane, nIdxBits);
    run(qstates.getNLanes(), 1, bitShiftMap, unaryGateFunc);
}

template<class real> void CPUQubitProcessor<real>::
applyControlGate(const Matrix2x2C64 &_mat, qgate::QubitStates &_qstates,
                 const qgate::IdList &localControlLanes, int localTargetLane) {

    CPUQubitStates<real> &qstates = static_cast<CPUQubitStates<real>&>(_qstates);
    Matrix2x2CR mat(_mat);
    
    /* control bit mask */
    QstateIdx allControlBits = qgate::createBitmask(localControlLanes);
    /* target bit */
    QstateIdx targetBit = Qone << localTargetLane;
    
    auto controlGateFunc =
            [allControlBits, targetBit, mat, &qstates](QstateIdx idx, QstateIdx) {
        QstateIdx idx_0 = idx | allControlBits;
        QstateIdx idx_1 = idx_0 | targetBit;
        const Complex qs0 = qstates[idx_0];
        const Complex qs1 = qstates[idx_1];
        Complex qsout0 = mat(0, 0) * qs0 + mat(0, 1) * qs1;
        Complex qsout1 = mat(1, 0) * qs0 + mat(1, 1) * qs1;
        qstates[idx_0] = qsout0;
        qstates[idx_1] = qsout1;
    };
    
    int nInputBits = (int)localControlLanes.size() + 1;
    int nIdxBits = qstates.getNLanes() - nInputBits;

    qgate::IdList allLanes(localControlLanes);
    allLanes.push_back(localTargetLane);
    qgate::IdList bitShiftMap = qgate::createBitShiftMap(allLanes, nIdxBits);
    run(qstates.getNLanes(), nInputBits, bitShiftMap, controlGateFunc);
}

template<class real>
void CPUQubitProcessor<real>::synchronize() {
}


template class CPUQubitProcessor<float>;
template class CPUQubitProcessor<double>;
