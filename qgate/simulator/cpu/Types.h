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



#include <stdarg.h>

#ifdef __GNUC__
#define FORMATATTR(stringIdx, firstToCheck) __attribute__((format(printf, stringIdx, firstToCheck)))
#else
#define FORMATATTR(stringIdx, firstToCheck)
#endif


namespace runtime {

void __abort(const char *file, unsigned long line);
void __abort(const char *file, unsigned long line, const char *format, ...) FORMATATTR(3, 4);

template <class... Args> inline
void _abort(const char *file, unsigned long line, Args... args) {
    __abort(file, line, args...);
}

void __throwError(const char *file, unsigned long line);
void __throwError(const char *file, unsigned long line, const char *format, ...) FORMATATTR(3, 4);

template <class... Args> inline
void _throwError(const char *file, unsigned long line, Args... args) {
    __throwError(file, line, args...);
}


void log(const char *format, ...) FORMATATTR(1, 2);

}

/* FIXME: undef somewhere */
#define abort_(...) runtime::_abort(__FILE__, __LINE__, __VA_ARGS__)
#define abortIf(cond, ...) if (cond) runtime::_abort(__FILE__, __LINE__, __VA_ARGS__)
#define throwError(...) runtime::_throwError(__FILE__, __LINE__, __VA_ARGS__)
#define throwErrorIf(cond, ...) if (cond) runtime::_throwError(__FILE__, __LINE__, __VA_ARGS__)


#ifndef _DEBUG
#  ifndef NDEBUG
#    define NDEBUG
#  endif
#endif

#include <assert.h> /* Here's the only place to include assert.h */
