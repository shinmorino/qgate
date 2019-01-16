#pragma once

#include <vector>
#include <complex>
#include <assert.h>
#include <stdarg.h>

namespace qgate {

typedef long long QstateIdx;
typedef long long QstateSize;
typedef std::vector<int> IdList;

struct QubitStates;
typedef std::vector<QubitStates*> QubitStatesList;

template<class real> using ComplexType = std::complex<real>;

template<class V, int D>
struct MatrixType {
    MatrixType() { }
    
    template<class Vsrc>
    explicit MatrixType(const MatrixType<Vsrc, D> &src) {
        for (int irow = 0; irow < D; ++irow)
            for (int icol = 0; icol < D; ++icol)
                elements_[irow][icol] = V(src(irow, icol));
    }
    
    V &operator()(int row, int col) {
        return elements_[row][col];
    }
    
    const V &operator()(int row, int col) const {
        return elements_[row][col];
    }

    V elements_[D][D];
};

/* Matrix for public interface. */
typedef MatrixType<ComplexType<double>, 2> Matrix2x2C64;


enum Precision {
    precUnknown = 0,
    precFP64,
    precFP32,
};

enum MathOp {
    mathOpNull = 0,
    mathOpProb,
};

const QstateIdx Qone = 1;
const QstateIdx Qtwo = 2;

/* round up, round down */
template<class V> inline
V roundDown(V value, int base) {
    return (value / base) * base;
}

template<class V> inline
V divru(V value, int base) {
    return (value + base - 1) / base;
}

template<class V> inline
V roundUp(V value, int base) {
    return ((value + base - 1) / base) * base;
}


#ifdef __GNUC__
#define FORMATATTR(stringIdx, firstToCheck) __attribute__((format(printf, stringIdx, firstToCheck)))
#else
#define FORMATATTR(stringIdx, firstToCheck)
#endif

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

/* FIXME: undef somewhere */
#define abort_(...) qgate::_abort(__FILE__, __LINE__, __VA_ARGS__)
#define abortIf(cond, ...) if (cond) qgate::_abort(__FILE__, __LINE__, __VA_ARGS__)
#define throwError(...) qgate::_throwError(__FILE__, __LINE__, __VA_ARGS__)
#define throwErrorIf(cond, ...) if (cond) qgate::_throwError(__FILE__, __LINE__, __VA_ARGS__)


#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

#ifndef _DEBUG
#  ifndef NDEBUG
#    define NDEBUG
#  endif
#endif


}


#include <assert.h> /* Here's the only place to include assert.h */
