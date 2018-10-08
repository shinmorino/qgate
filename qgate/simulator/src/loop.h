#pragma once

#include "Types.h"
#include <functional>

template<class C>
void for_(QstateIdxType begin, QstateIdxType end, const C &functor) {
    std::function<void(QstateIdxType)> func = std::move(functor);
#ifdef _OPENMP
    if ((1 << 16) < end - begin) { 
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

template<class C>
real sum(QstateIdxType begin, QstateIdxType end, const C &functor) {
    std::function<real(QstateIdxType)> func = std::move(functor);
    real v = real(0.);
#ifdef _OPENMP
    if ((1 << 16) < end - begin) { 
#pragma omp parallel for
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
