#include "DeviceSum.h"

using namespace qgate_cuda;

template<class V>
DeviceSumType<V>::DeviceSumType() {
    h_partialSum_ = NULL;
    nBlocks_ = -1;
}

template<class V>
DeviceSumType<V>::~DeviceSumType() {
}

template<class V>
void DeviceSumType<V>::prepare() {
    int devNo;
    cudaDeviceProp prop;
    throwOnError(cudaGetDevice(&devNo));
    throwOnError(cudaGetDeviceProperties(&prop, devNo));
    nBlocks_ = prop.multiProcessorCount * (2048 / 128) * 4;
}

template<class V>
void DeviceSumType<V>::allocate(CUDAResource &rsrc) {
    assert(h_partialSum_ == NULL);
    h_partialSum_ = rsrc.getHostMem<V>(sizeof(V) * nBlocks_);
}

template<class V>
void DeviceSumType<V>::deallocate(CUDAResource &rsrc) {
    h_partialSum_ = NULL;
}


template DeviceSumType<float>;
template DeviceSumType<double>;