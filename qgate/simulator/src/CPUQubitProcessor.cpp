#include "CPUQubitProcessor.h"
#include "CPUQubitStates.h"
#include <algorithm>
#include <string.h>

using namespace qgate_cpu;
using qgate::Qone;
using qgate::Qtwo;

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

template<class real>
void CPUQubitProcessor<real>::initializeQubitStates(const qgate::IdList &qregIdList, qgate::QubitStates &_qstates,
                                                    int nLanesPerDevice, qgate::IdList &_deviceIds) {
    CPUQubitStates<real> &qstates = static_cast<CPUQubitStates<real>&>(_qstates);
    qstates.allocate(qregIdList);
}

template<class real>
void CPUQubitProcessor<real>::resetQubitStates(qgate::QubitStates &_qstates) {
    CPUQubitStates<real> &qstates = static_cast<CPUQubitStates<real>&>(_qstates);
    
    Complex *cmp = qstates.getPtr();
    qgate::QstateSize nStates = Qone << qstates.getNQregs();

    auto setZeroFunc = [=](int threadIdx, QstateIdx spanBegin, QstateIdx spanEnd) {
        memset(&cmp[spanBegin], 0, sizeof(Complex) * (spanEnd - spanBegin));
    };
    parallel_.distribute(0LL, nStates, setZeroFunc);
    cmp[0] = Complex(1.);
}

template<class real> double CPUQubitProcessor<real>::
calcProbability(const qgate::QubitStates &_qstates, int qregId) {
    const CPUQubitStates<real> &qstates = static_cast<const CPUQubitStates<real>&>(_qstates);
    int lane = qstates.getLane(qregId);
    return _calcProbability(qstates, lane);
}

template<class real>
real CPUQubitProcessor<real>::_calcProbability(const CPUQubitStates<real> &qstates, int lane) {

    QstateIdx bitmask_hi = ~((Qtwo << lane) - 1);
    QstateIdx bitmask_lo = (Qone << lane) - 1;
    QstateIdx nStates = Qone << (qstates.getNQregs() - 1);
    
    auto sumFunc = [=, &qstates](QstateIdx idx) {
        QstateIdx idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
        const Complex &qs = qstates[idx_lo];
        return abs2<real>(qs);
    };
    real prob = parallel_.sum<real>(0, nStates, sumFunc);
    return prob;
}

template<class real>
int CPUQubitProcessor<real>::measure(double randNum, qgate::QubitStates &_qstates, int qregId) {
    
    CPUQubitStates<real> &qstates = static_cast<CPUQubitStates<real>&>(_qstates);

    int lane = qstates.getLane(qregId);
    real prob = (real)_calcProbability(qstates, lane);
    
    int cregValue = -1;
    
    QstateIdx bitmask_lane = Qone << lane;
    QstateIdx bitmask_hi = ~((Qtwo << lane) - 1);
    QstateIdx bitmask_lo = (Qone << lane) - 1;
    QstateIdx nStates = Qone << (qstates.getNQregs() - 1);

    if (real(randNum) < prob) {
        cregValue = 0;
        real norm = real(1.) / std::sqrt(prob);

        auto fmeasure_0 = [=, &qstates](QstateIdx idx) {
            QstateIdx idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
            QstateIdx idx_hi = idx_lo | bitmask_lane;
            qstates[idx_lo] *= norm;
            qstates[idx_hi] = real(0.);
        };
        parallel_.for_each(0, nStates, fmeasure_0);
    }
    else {
        cregValue = 1;
        real norm = real(1.) / std::sqrt(real(1.) - prob);
        auto fmeasure_1 = [=, &qstates](QstateIdx idx) {
            QstateIdx idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
            QstateIdx idx_hi = idx_lo | bitmask_lane;
            qstates[idx_lo] = real(0.);
            qstates[idx_hi] *= norm;
        };
        parallel_.for_each(0, nStates, fmeasure_1);
    }
    return cregValue;
}
    

template<class real>
void CPUQubitProcessor<real>::applyReset(qgate::QubitStates &_qstates, int qregId) {

    CPUQubitStates<real> &qstates = static_cast<CPUQubitStates<real>&>(_qstates);
    
    int lane = qstates.getLane(qregId);
    
    QstateIdx bitmask_lane = Qone << lane;
    QstateIdx bitmask_hi = ~((Qtwo << lane) - 1);
    QstateIdx bitmask_lo = (Qone << lane) - 1;
    QstateIdx nStates = Qone << (qstates.getNQregs() - 1);
    
    /* Assuming reset is able to be applyed after measurement.
     * Ref: https://quantumcomputing.stackexchange.com/questions/3908/possibility-of-a-reset-quantum-gate */
    auto freset = [=, &qstates](QstateIdx idx) {
        QstateIdx idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
        QstateIdx idx_hi = idx_lo | bitmask_lane;
        qstates[idx_lo] = qstates[idx_hi];
        qstates[idx_hi] = real(0.);
    };
    parallel_.for_each(0, nStates, freset);
}

template<class real>
void CPUQubitProcessor<real>::applyUnaryGate(const Matrix2x2C64 &_mat, qgate::QubitStates &_qstates, int qregId) {
    
    CPUQubitStates<real> &qstates = static_cast<CPUQubitStates<real>&>(_qstates);
    Matrix2x2CR mat(_mat);
    
    int lane = qstates.getLane(qregId);
    
    QstateIdx bitmask_lane = Qone << lane;
    QstateIdx bitmask_hi = ~((Qtwo << lane) - 1);
    QstateIdx bitmask_lo = (Qone << lane) - 1;
    QstateIdx nStates = Qone << (qstates.getNQregs() - 1);
    for (QstateIdx idx = 0; idx < nStates; ++idx) {
        QstateIdx idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo);
        QstateIdx idx_hi = idx_lo | bitmask_lane;
        const Complex &qs0 = qstates[idx_lo];
        const Complex &qs1 = qstates[idx_hi];
        Complex qsout0 = mat(0, 0) * qs0 + mat(0, 1) * qs1;
        Complex qsout1 = mat(1, 0) * qs0 + mat(1, 1) * qs1;
        qstates[idx_lo] = qsout0;
        qstates[idx_hi] = qsout1;
    }
}

template<class real>
void CPUQubitProcessor<real>::applyControlGate(const Matrix2x2C64 &_mat, qgate::QubitStates &_qstates, int controlId, int targetId) {
    
    CPUQubitStates<real> &qstates = static_cast<CPUQubitStates<real>&>(_qstates);
    Matrix2x2CR mat(_mat);
    
    int lane0 = qstates.getLane(controlId);
    int lane1 = qstates.getLane(targetId);
    QstateIdx bitmask_control = Qone << lane0;
    QstateIdx bitmask_target = Qone << lane1;
    
    QstateIdx bitmask_lane_max = std::max(bitmask_control, bitmask_target);
    QstateIdx bitmask_lane_min = std::min(bitmask_control, bitmask_target);
    
    QstateIdx bitmask_hi = ~(bitmask_lane_max * 2 - 1);
    QstateIdx bitmask_mid = (bitmask_lane_max - 1) & ~((bitmask_lane_min << 1) - 1);
    QstateIdx bitmask_lo = bitmask_lane_min - 1;
    
    QstateIdx nStates = Qone << (qstates.getNQregs() - 2);

    auto fcg = [=, &qstates](QstateIdx idx) {
        QstateIdx idx_0 = ((idx << 2) & bitmask_hi) | ((idx << 1) & bitmask_mid) | (idx & bitmask_lo) | bitmask_control;
        QstateIdx idx_1 = idx_0 | bitmask_target;
        
        const Complex &qs0 = qstates[idx_0];
        const Complex &qs1 = qstates[idx_1];;
        Complex qsout0 = mat(0, 0) * qs0 + mat(0, 1) * qs1;
        Complex qsout1 = mat(1, 0) * qs0 + mat(1, 1) * qs1;
        qstates[idx_0] = qsout0;
        qstates[idx_1] = qsout1;
    };    
    parallel_.for_each(0, nStates, fcg);
}


template<class real> template<class R, class F>
void CPUQubitProcessor<real>::qubitsGetValues(R *values, const F &func,
                                              const QubitStatesList &qstatesList,
                                              QstateIdx beginIdx, QstateIdx endIdx) {
    int nQubitStates = (int)qstatesList.size();
    const CPUQubitStates<real> **qstates = new const CPUQubitStates<real>*[nQubitStates];
    
    for (int idx = 0; idx < nQubitStates; ++idx)
        qstates[idx] = static_cast<const CPUQubitStates<real>*>(qstatesList[idx]);
    
    auto fgetstates = [=, &qstates, &func](QstateIdx idx) {
        R v = R(1.);
        for (int qstatesIdx = 0; (int)qstatesIdx < nQubitStates; ++qstatesIdx) {
            const ComplexType<real> &state = qstates[qstatesIdx]->getStateByQregIdx(idx);
            v *= func(state);
        }
        values[idx] = v;
    };

    auto fgetstates256 = [=, &func](QstateIdx idx, QstateIdx spanBegin, QstateIdx spanEnd) {
        Complex states[256];
        for (QstateIdx idx256 = spanBegin; idx256 < spanEnd; idx256 += 256) {
            R *v = &values[idx256];
            for (int idx = 0; idx < 256; ++idx)
                v[idx] = R(1.);
            for (int qstatesIdx = 0; (int)qstatesIdx < nQubitStates; ++qstatesIdx) {
                qstates[qstatesIdx]->getStateByQregIdx256(states, idx256);
                for (int idx = 0; idx < 256; ++idx)
                    v[idx] *= func(states[idx]);
            }
        }
    };

    if (endIdx - beginIdx < 256) {
        parallel_.for_each(beginIdx, endIdx, fgetstates, 1);
    }
    else {
        using qgate::roundDown;
        using qgate::roundUp;

        if ((beginIdx % 256) != 0) {
            QstateIdx endIdx256 = roundUp(beginIdx, 256LL);
            parallel_.for_each(beginIdx, endIdx256, fgetstates);
        }
        QstateIdx beginIdx256 = roundUp(beginIdx, 256LL);
        QstateIdx endIdx256 = roundDown(endIdx, 256LL);
        qgate::Parallel(-1, 256).distribute(beginIdx256, endIdx256, fgetstates256);

        if ((endIdx % 256) != 0) {
            QstateIdx endIdx256 = roundDown(beginIdx, 256LL);
            parallel_.for_each(endIdx256, endIdx, fgetstates, 1);
        }
    }

    delete[] qstates;
}

template<class real>
void CPUQubitProcessor<real>::getStates(void *array, QstateIdx arrayOffset,
                                        MathOp op,
                                        const QubitStatesList &qstatesList,
                                        QstateIdx beginIdx, QstateIdx endIdx) {
    
    const qgate::QubitStates *qstates = qstatesList[0];
    if (sizeof(real) == sizeof(float))
        abortIf(qstates->getPrec() != qgate::precFP32, "Wrong type");
    if (sizeof(real) == sizeof(double))
        abortIf(qstates->getPrec() != qgate::precFP64, "Wrong type");

    switch (op) {
    case qgate::mathOpNull: {
        ComplexType<real> *cmpArray = static_cast<ComplexType<real>*>(array);
        qubitsGetValues(&cmpArray[arrayOffset], null<real>,
                        qstatesList, beginIdx, endIdx);
        break;
    }
    case qgate::mathOpProb: {
        real *vArray = static_cast<real*>(array);
        qubitsGetValues(&vArray[arrayOffset], abs2<real>,
                        qstatesList, beginIdx, endIdx);
        break;
    }
    default:
        abort_("Unknown math op.");
    }
}


template class CPUQubitProcessor<float>;
template class CPUQubitProcessor<double>;
