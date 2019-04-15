/* -*- c++ -*- */
#pragma once

#include "Types.h"
#include <cuda_runtime.h>


namespace qgate_cuda {

enum {
    MAX_N_CHUNKS = 32
};


template<class real>
struct DeviceComplexType;

template<>
struct __align__(8) DeviceComplexType<float> {
    /* FIXME: rename */
    float real;
    float imag;
    
    __device__ __host__
    DeviceComplexType() { }
    
    __host__
    DeviceComplexType(const qgate::ComplexType<float> &c) : real(c.real()), imag(c.imag()) { }
    
    __host__
    DeviceComplexType(const qgate::ComplexType<double> &c) : real((float)c.real()), imag((float)c.imag()) { }
    
    __host__ __device__
    DeviceComplexType(const DeviceComplexType<float> &c) : real(c.real), imag(c.imag) { }
    
    __device__ __host__
    DeviceComplexType(float _real, float _imag = 0.f) : real(_real), imag(_imag) { }
    
};

template<>
struct __align__(16) DeviceComplexType<double> {
    double real;
    double imag;
    
    __device__ __host__
    DeviceComplexType() { }
    
    __host__
    DeviceComplexType(const qgate::ComplexType<double> &c) : real(c.real()), imag(c.imag()) { }
    
    __host__ __device__
    DeviceComplexType(const DeviceComplexType<double> &c) : real(c.real), imag(c.imag) { }
    
    __device__ __host__
    DeviceComplexType(double _real, double _imag = 0.) : real(_real), imag(_imag) { }

};


template<class real>
__device__ __forceinline__
const DeviceComplexType<real> &operator*=(DeviceComplexType<real> &dc, real v) {
    dc.real *= v;
    dc.imag *= v;
    return dc;
}

template<class real>
__device__ __forceinline__
DeviceComplexType<real> operator+(const DeviceComplexType<real> &c0, const DeviceComplexType<real> &c1) {
    real re = c0.real + c1.real;
    real im = c0.imag + c1.imag;
    return DeviceComplexType<real>(re, im);
}

template<class real>
__device__ __forceinline__
DeviceComplexType<real> operator*(const DeviceComplexType<real> &c0, const DeviceComplexType<real> &c1) {
    real re = c0.real * c1.real - c0.imag * c1.imag;
    real im = c0.real * c1.imag + c0.imag * c1.real;
    return DeviceComplexType<real>(re, im);
}

template<class real>
__device__ __forceinline__
DeviceComplexType<real> operator*(const real &c0, const DeviceComplexType<real> &c1) {
    real re = c0 * c1.real;
    real im = c0 * c1.imag;
    return DeviceComplexType<real>(re, im);
}

template<class real>
__device__ __forceinline__
const DeviceComplexType<real> &operator*=(DeviceComplexType<real> &c0, const DeviceComplexType<real> &c1) {
    DeviceComplexType<real> prod = c0 * c1;
    c0 = prod;
    return c0;
}

template<class V, int D>
struct DeviceMatrixType {
    enum { _D = D };
    
    template<class VH>
    DeviceMatrixType(const qgate::MatrixType<VH, D> &hostMatrix) {
        for (int row = 0; row < D; ++row) {
            for (int col = 0; col < D; ++col) {
                elements_[row][col] = V(hostMatrix(row, col));
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

template<class R>
using DeviceMatrix2x2C = DeviceMatrixType<DeviceComplexType<R>, 2> ;


/* FIXME: undef somewhere. */
#ifdef _DEBUG
#define DEBUG_SYNC {throwOnError(cudaGetLastError()); throwOnError(cudaDeviceSynchronize()); }
#else
#define DEBUG_SYNC
#endif


inline bool _valid(cudaError_t cuerr) { return cuerr == cudaSuccess; }
void _throwError(cudaError_t status, const char *file, unsigned long line, const char *expr);

#define throwOnError(expr) { auto status = (expr); if (!qgate_cuda::_valid(status)) { qgate_cuda::_throwError(status, __FILE__, __LINE__, #expr); } }

using qgate::divru;
using qgate::roundUp;

}
