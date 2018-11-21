#pragma once

#include "Types.h"
#include <functional>

namespace qgate_cpu {

using qgate::QstateIdxType;

template<class C>
void parallel_for_each(QstateIdxType begin, QstateIdxType end, const C &functor) {
    throwErrorIf(0x40000000LL < end, "end < 0x40000000LL");

    std::function<void(QstateIdxType)> func = std::move(functor);
#ifdef _OPENMP
    if ((1LL << 16) < end - begin) {
#pragma omp parallel for
        for (QstateIdxType idx = begin; idx < end; ++idx) {
            func(idx);
        }
    }  else
#endif
    for (QstateIdxType idx = begin; idx < end; ++idx) {
        func(idx);
    }
};

template<class real, class C>
real sum(QstateIdxType begin, QstateIdxType end, const C &functor) {
    throwErrorIf(0x40000000LL < end, "end < 0x40000000LL");

    std::function<real(QstateIdxType)> func = std::move(functor);
    real v = real(0.);
#ifdef _OPENMP
    if ((1LL << 16) < end - begin) { 
#pragma omp parallel for reduction(+:v)
        for (QstateIdxType idx = begin; idx < end; ++idx) {
            v += func(idx);
        }
    }  else
#endif
    for (QstateIdxType idx = begin; idx < end; ++idx) {
        v += func(idx);
    }
    
    return v;
};

}
