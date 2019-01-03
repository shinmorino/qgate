#pragma once
#include "DeviceTypes.h"

namespace qgate_cuda {

template<class R>
struct abs2 {
    __device__ __forceinline__
    R operator()(const DeviceComplexType<R> &c) const {
        return c.real * c.real + c.imag * c.imag;
    }
};

template<class V>
struct null {
    __device__ __forceinline__
    const DeviceComplexType<V> &operator()(const DeviceComplexType<V> &c) const {
        return c;
    }
};

}
