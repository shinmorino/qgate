#pragma once

#include "Types.h"
#include <functional>

namespace qgate_cpu {

using qgate::QstateIdx;

template<class C>
void parallel_for_each(QstateIdx begin, QstateIdx end, const C &functor) {
    throwErrorIf(0x40000000LL < end, "end < 0x40000000LL");

    std::function<void(QstateIdx)> func = std::move(functor);
#ifdef _OPENMP
    if ((1LL << 16) < end - begin) {
#pragma omp parallel for
        for (QstateIdx idx = begin; idx < end; ++idx) {
            func(idx);
        }
    }  else
#endif
    for (QstateIdx idx = begin; idx < end; ++idx) {
        func(idx);
    }
};

template<class real, class C>
real sum(QstateIdx begin, QstateIdx end, const C &functor) {
    throwErrorIf(0x40000000LL < end, "end < 0x40000000LL");

    std::function<real(QstateIdx)> func = std::move(functor);
    real v = real(0.);
#ifdef _OPENMP
    if ((1LL << 16) < end - begin) { 
#pragma omp parallel for reduction(+:v)
        for (QstateIdx idx = begin; idx < end; ++idx) {
            v += func(idx);
        }
    }  else
#endif
    for (QstateIdx idx = begin; idx < end; ++idx) {
        v += func(idx);
    }
    
    return v;
};

}
