#pragma once

#include "Types.h"
#include <cuda_runtime.h>


namespace cuda_runtime {

struct __align__(8) DeviceComplex {
    /* FIXME: rename */
    float re;
    float im;
    
    __device__ __host__
    DeviceComplex() { }
    
    __host__
    DeviceComplex(const Complex &c) : re(c.real()), im(c.imag()) { }
    
    __host__ __device__
    DeviceComplex(const DeviceComplex &c) : re(c.re), im(c.im) { }
    
    __device__ __host__
    DeviceComplex(real _re, real _im = float(0.)) : re(_re), im(_im) { }
    
};

__device__ __forceinline__
const DeviceComplex &operator*=(DeviceComplex &dc, real v) {
    dc.re *= v;
    dc.im *= v;
    return dc;
}

__device__ __forceinline__
DeviceComplex operator+(const DeviceComplex &c0, const DeviceComplex &c1) {
    real re = c0.re + c1.re;
    real im = c0.im * c1.im;
    return DeviceComplex(re, im);
}

__device__ __forceinline__
DeviceComplex operator*(const DeviceComplex &c0, const DeviceComplex &c1) {
    real re = c0.re * c1.re - c0.im * c1.im;
    real im = c0.re * c1.im + c0.im * c1.re;
    return DeviceComplex(re, im);
}

__device__ __forceinline__
const DeviceComplex &operator*=(DeviceComplex &c0, const DeviceComplex &c1) {
    DeviceComplex prod = c0 * c1;
    c0 = prod;
    return c0;
}


template<class V, int D>
struct DeviceMatrix {
    enum { _D = D };
    
    template<class VH>
    DeviceMatrix(const Matrix<VH, D> &hostMatrix) {
        for (int row = 0; row < D; ++row) {
            for (int col = 0; col < D; ++col) {
                elements_[row][col] = hostMatrix(row, col);
            }
        }
    }
    
    __device__ __forceinline__
    V &operator()(int row, int col) {
        return elements_[row][col];
    }

    __device__ __forceinline__
    const V &operator()(int row, int col) const {
        return elements_[row][col];
    }

    V elements_[D][D];
};

typedef DeviceMatrix<DeviceComplex, 2> DeviceCMatrix2x2;



/* FIXME: undef somewhere. */
#ifdef _DEBUG
#define DEBUG_SYNC {throwOnError(cudaGetLastError()); throwOnError(cudaDeviceSynchronize()); }
#else
#define DEBUG_SYNC
#endif


inline bool _valid(cudaError_t cuerr) { return cuerr == cudaSuccess; }
void _throwError(cudaError_t status, const char *file, unsigned long line, const char *expr);

#define throwOnError(expr) { auto status = (expr); if (!cuda_runtime::_valid(status)) { cuda_runtime::_throwError(status, __FILE__, __LINE__, #expr); } }

template<class V>
inline V divru(V v, int base) {
    return (v + base - 1) / base;
}
        
template<class V>
inline V roundUp(V v, int base) {
    return divru(v, base) * base;
}


}
