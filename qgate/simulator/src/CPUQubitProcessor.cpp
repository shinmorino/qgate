#include "CPUQubitProcessor.h"
#include "CPUQubitStates.h"
#include <algorithm>
#include <string.h>
#include "BitPermTable.h"
#include "Parallel.h"
#include <valarray>

using namespace qgate_cpu;
using qgate::Qone;
using qgate::Qtwo;
using qgate::Parallel;

namespace {

template<class R>
inline R abs2(const std::complex<R> &c) {
    return c.real() * c.real() + c.imag() * c.imag();
}

template<class R>
inline std::complex<R> null(const std::complex<R> &c) {
    return c;
}

}

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
        qgate::Parallel(-1, 256).distribute(0LL, nLoops, gateFunc256);
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
    Parallel(-1).distribute(0LL, nStates, setZeroFunc);
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
        return abs2<real>(qs);
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
        qgate::Parallel parallel(-1, 256);
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
        parallel.distribute(0LL, nLoops, forloop);
        prob = real(0.);
        /* FIXME: when (end - begin) is small, actual nWorkers is 1, though nWorkers is used here. */
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
        qgate::Parallel().distribute(0LL, dstSize, copyAndZeroFunc);
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
    qgate::Parallel().distribute(0LL, Nsrc1, kronFunc);
    
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
        qgate::Parallel().distribute(0LL, productSize, kronInPlaceF_0);
        
        Complex vSrc = src[0];
        auto kronInPlaceF_1 = [=](int threadIdx, QstateIdx spanBegin, QstateIdx spanEnd) {
            for (QstateIdx idx = spanBegin; idx < spanEnd; ++idx)
                dst[idx] *= vSrc;
        };
        qgate::Parallel().distribute(0LL, productSize, kronInPlaceF_1);

        productSize *= Nsrc;
    }

    /* zero fill */
    auto zero = [=](int threadIdx, QstateIdx spanBegin, QstateIdx spanEnd) {
        QstateSize zeroSize = spanEnd - spanBegin;
        memset(&dst[spanBegin], 0, sizeof(Complex) * zeroSize);
    };
    qgate::Parallel().distribute(productSize, dstSize, zero);
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


template<class real> template<class R, class F> void CPUQubitProcessor<real>::
qubitsGetValues(R *values, const F &func,
                const qgate::IdList *laneTransTables, QstateIdx emptyLaneMask,
                const QubitStatesList &qstatesList,
                QstateSize nStates, QstateIdx begin, QstateIdx step) {
    
    int nQubitStates = (int)qstatesList.size();
    /* array of CPUQubitStates */
    const CPUQubitStates<real> **qstates = new const CPUQubitStates<real>*[nQubitStates];
    for (int idx = 0; idx < nQubitStates; ++idx)
        qstates[idx] = static_cast<const CPUQubitStates<real>*>(qstatesList[idx]);
    /* array of BitPermTable. */
    qgate::BitPermTable *perm = new qgate::BitPermTable[nQubitStates];
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
        Parallel(1).for_each(0, nStates, fgetstates);
    }
    else if (256 < step) {
        Parallel(-1).for_each(0, nStates, fgetstates);
    }
    else {
        QstateSize nStates256 = qgate::roundDown(nStates, 256);
        qgate::Parallel(-1, 256).distribute(0LL, nStates256, fgetstates256);
        if (nStates256 != nStates)
            Parallel(-1).for_each(nStates256, nStates, fgetstates);
    }

    delete[] perm;
    delete[] qstates;
}

template<class real>
void CPUQubitProcessor<real>::
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
        qubitsGetValues(&cmpArray[arrayOffset], null<real>,
                        laneTransTables, emptyLaneMask, qstatesList, nStates, begin, step);
        break;
    }
    case qgate::mathOpProb: {
        real *vArray = static_cast<real*>(array);
        qubitsGetValues(&vArray[arrayOffset], abs2<real>,
                        laneTransTables, emptyLaneMask, qstatesList, nStates, begin, step);
        break;
    }
    default:
        abort_("Unknown math op.");
    }
}


template class CPUQubitProcessor<float>;
template class CPUQubitProcessor<double>;
