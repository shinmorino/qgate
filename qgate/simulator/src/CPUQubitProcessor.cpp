#include "CPUQubitProcessor.h"
#include "CPUQubitStates.h"
#include <algorithm>
#include <string.h>
#include "BitPermTable.h"
#include "Parallel.h"

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
void CPUQubitProcessor<real>::clear() {
}

template<class real>
void CPUQubitProcessor<real>::prepare() {
}

template<class real> void CPUQubitProcessor<real>::
initializeQubitStates(qgate::QubitStates &_qstates,
                      int nLanes, int nLanesPerDevice, qgate::IdList &_deviceIds) {
    CPUQubitStates<real> &qstates = static_cast<CPUQubitStates<real>&>(_qstates);
    qstates.allocate(nLanes);
}

template<class real> template<class P, class G>
void CPUQubitProcessor<real>::run(CPUQubitStates<real> &qstates, int nInputBits, const P &permf, const G &gatef) {

    int nLanes = (int)qstates.getNLanes();
    int nIdxBits = nLanes - nInputBits;
    qgate::BitPermTable perm;
    perm.init_idxToQstateIdx(nIdxBits, permf);

    QstateIdx nLoops = Qone << nIdxBits;
    if (nLoops < 256) {
        for (int idx = 0; idx < nLoops; ++idx) {
            QstateIdx idx_base = perm.permute_8bits(0, idx);
            gatef(idx_base);
        }
    }
    else {
        auto gateFunc256 =
            [=, &perm, &qstates](int threadIdx, QstateIdx spanBegin, QstateIdx spanEnd) {
            for (QstateIdx idx256 = spanBegin; idx256 < spanEnd; idx256 += 256) {
                QstateIdx idx56bits = perm.permute_56bits(idx256);
                for (int idx = 0; idx < 256; ++idx) {
                    QstateIdx idx_base = perm.permute_8bits(idx56bits, idx);
                    gatef(idx_base);
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

    QstateIdx bitmask_hi = ~((Qtwo << localLane) - 1);
    QstateIdx bitmask_lo = (Qone << localLane) - 1;
    
    auto sumFunc = [=, &qstates](QstateIdx idx_lo) {
        const Complex &qs = qstates[idx_lo];
        return abs2<real>(qs);
    };
    
    qgate::BitPermTable perm;
    auto permf = [=](QstateIdx bit) {
        return ((bit << 1) & bitmask_hi) | (bit & bitmask_lo);
    };
    int nLanes = qstates.getNLanes();
    perm.init_idxToQstateIdx(nLanes - 1, permf);
    
    QstateIdx nLoops = Qone << (nLanes - 1);
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
int CPUQubitProcessor<real>::measure(double randNum, qgate::QubitStates &_qstates, int localLane) {
    
    CPUQubitStates<real> &qstates = static_cast<CPUQubitStates<real>&>(_qstates);

    real prob = (real)_calcProbability(qstates, localLane);
    
    int cregValue = -1;
    
    QstateIdx bitmask_lane = Qone << localLane;
    QstateIdx bitmask_hi = ~((Qtwo << localLane) - 1);
    QstateIdx bitmask_lo = (Qone << localLane) - 1;

    std::function<void(QstateIdx)> fmeasure;

    if (real(randNum) < prob) {
        cregValue = 0;
        real norm = real(1.) / std::sqrt(prob);
        fmeasure = [=, &qstates](QstateIdx idx_lo) {
            QstateIdx idx_hi = idx_lo | bitmask_lane;
            qstates[idx_lo] *= norm;
            qstates[idx_hi] = real(0.);
        };
    }
    else {
        cregValue = 1;
        real norm = real(1.) / std::sqrt(real(1.) - prob);
        fmeasure = [=, &qstates](QstateIdx idx_lo) {
            QstateIdx idx_hi = idx_lo | bitmask_lane;
            qstates[idx_lo] = real(0.);
            qstates[idx_hi] *= norm;
        };
    }

    auto permf = [=](QstateIdx bit) {
        return ((bit << 1) & bitmask_hi) | (bit & bitmask_lo);
    };
    run(qstates, 1, permf, fmeasure);

    return cregValue;
}
    

template<class real>
void CPUQubitProcessor<real>::applyReset(qgate::QubitStates &_qstates, int localLane) {

    CPUQubitStates<real> &qstates = static_cast<CPUQubitStates<real>&>(_qstates);
    
    QstateIdx bitmask_lane = Qone << localLane;
    QstateIdx bitmask_hi = ~((Qtwo << localLane) - 1);
    QstateIdx bitmask_lo = (Qone << localLane) - 1;
    
    /* Assuming reset is able to be applyed after measurement.
     * Ref: https://quantumcomputing.stackexchange.com/questions/3908/possibility-of-a-reset-quantum-gate */

    auto resetFunc = [=, &qstates](QstateIdx idx_lo) {
                         QstateIdx idx_hi = idx_lo | bitmask_lane;
                         qstates[idx_lo] = qstates[idx_hi];
                         qstates[idx_hi] = real(0.);
                     };
    
    auto permf = [=](QstateIdx bit) {
        return ((bit << 1) & bitmask_hi) | (bit & bitmask_lo);
    };

    run(qstates, 1, permf, resetFunc);
}

template<class real>
void CPUQubitProcessor<real>::applyUnaryGate(const Matrix2x2C64 &_mat,
                                             qgate::QubitStates &_qstates, int localLane) {
    
    CPUQubitStates<real> &qstates = static_cast<CPUQubitStates<real>&>(_qstates);
    Matrix2x2CR mat(_mat);
    
    QstateIdx bitmask_lane = Qone << localLane;
    QstateIdx bitmask_hi = ~((Qtwo << localLane) - 1);
    QstateIdx bitmask_lo = (Qone << localLane) - 1;

    auto unaryGateFunc = [=, &qstates](QstateIdx idx_lo) {
                             QstateIdx idx_hi = idx_lo | bitmask_lane;
                             const Complex &qs0 = qstates[idx_lo];
                             const Complex &qs1 = qstates[idx_hi];
                             Complex qsout0 = mat(0, 0) * qs0 + mat(0, 1) * qs1;
                             Complex qsout1 = mat(1, 0) * qs0 + mat(1, 1) * qs1;
                             qstates[idx_lo] = qsout0;
                             qstates[idx_hi] = qsout1;
                         };

    auto permf = [=](QstateIdx bit) {
        return ((bit << 1) & bitmask_hi) | (bit & bitmask_lo);
    };

    run(qstates, 1, permf, unaryGateFunc);
}

template<class real> void CPUQubitProcessor<real>::
applyControlGate(const Matrix2x2C64 &_mat, qgate::QubitStates &_qstates,
                 int localControlLane, int localTargetLane) {

    CPUQubitStates<real> &qstates = static_cast<CPUQubitStates<real>&>(_qstates);
    Matrix2x2CR mat(_mat);

    QstateIdx bit_control = Qone << localControlLane;
    QstateIdx bit_target = Qone << localTargetLane;

    QstateIdx bitmask_lane_max = std::max(bit_control, bit_target);
    QstateIdx bitmask_lane_min = std::min(bit_control, bit_target);

    QstateIdx bitmask_hi = ~(bitmask_lane_max * 2 - 1);
    QstateIdx bitmask_mid = (bitmask_lane_max - 1) & ~((bitmask_lane_min << 1) - 1);
    QstateIdx bitmask_lo = bitmask_lane_min - 1;

    auto controlGateFunc = [=, &qstates](QstateIdx idx) {
                               QstateIdx idx_0 = idx | bit_control;
                               QstateIdx idx_1 = idx_0 | bit_target;
                               const Complex qs0 = qstates[idx_0];
                               const Complex qs1 = qstates[idx_1];
                               Complex qsout0 = mat(0, 0) * qs0 + mat(0, 1) * qs1;
                               Complex qsout1 = mat(1, 0) * qs0 + mat(1, 1) * qs1;
                               qstates[idx_0] = qsout0;
                               qstates[idx_1] = qsout1;
                           };
    
    auto permf = [=](QstateIdx bit) {
        return ((bit << 2) & bitmask_hi) | ((bit << 1) & bitmask_mid) | (bit & bitmask_lo);
    };
    run(qstates, 2, permf, controlGateFunc);
}


template<class real> template<class R, class F> void CPUQubitProcessor<real>::
qubitsGetValues(R *values, const F &func,
                const qgate::IdList *laneTransTables, const QubitStatesList &qstatesList,
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
        for (int qStatesIdx = 0; qStatesIdx < nQubitStates; ++qStatesIdx) {
            int localIdx = perm[qStatesIdx].permute(extSrcIdx);
            const ComplexType<real> &state = qstates[qStatesIdx]->operator[](localIdx);
            v *= func(state);
        }
        values[extDstIdx] = v;
    };

    auto fgetstates256 = [=, &func](QstateIdx idx, QstateIdx spanBegin, QstateIdx spanEnd) {
        /* spanBegin and spanEnd are multiples of 256. */
        QstateIdx extSrcIdx = begin + step * spanBegin;
        QstateIdx extSrcIdx_prefix = qgate::roundDown(extSrcIdx, 256);
        int extSrcIdx_8bits = (int)(extSrcIdx % 256);

        enum { loopStep = 256 };
        for (QstateIdx dstIdx256 = spanBegin; dstIdx256 < spanEnd; dstIdx256 += loopStep) {
            R *v = &values[dstIdx256];
            for (int idx = 0; idx < loopStep; ++idx)
                v[idx] = R(1.);
            for (int qstatesIdx = 0; qstatesIdx < nQubitStates; ++qstatesIdx) {
                QstateIdx extSrcIdx_prefix_tmp = extSrcIdx_prefix;
                int extSrcIdx_8bits_tmp = extSrcIdx_8bits;
                QstateIdx cached = perm[qstatesIdx].permute_56bits(extSrcIdx_prefix);

                for (int idx = 0; idx < loopStep; ++idx) {
                    QstateIdx localIdx = perm[qstatesIdx].permute_8bits(cached, extSrcIdx_8bits_tmp);
                    const ComplexType<real> &state = qstates[qstatesIdx]->operator[](localIdx);
                    v[idx] *= func(state);

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
          const qgate::IdList *laneTransTables, const QubitStatesList &qstatesList,
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
                        laneTransTables, qstatesList, nStates, begin, step);
        break;
    }
    case qgate::mathOpProb: {
        real *vArray = static_cast<real*>(array);
        qubitsGetValues(&vArray[arrayOffset], abs2<real>,
                        laneTransTables, qstatesList, nStates, begin, step);
        break;
    }
    default:
        abort_("Unknown math op.");
    }
}


template class CPUQubitProcessor<float>;
template class CPUQubitProcessor<double>;
