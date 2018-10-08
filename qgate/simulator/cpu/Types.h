#pragma once

#include <vector>
#include <complex>
#include <assert.h>


typedef float real;
typedef unsigned long long QstateIdxType;
typedef std::complex<real> Complex;
typedef std::vector<int> IdList;


template<class V, int D>
struct Matrix {
    V &operator()(int row, int col) {
        return elements_[row][col];
    }
    
    const V &operator()(int row, int col) const {
        return elements_[row][col];
    }

    V elements_[D][D];
};

typedef Matrix<Complex, 2> CMatrix2x2;


