#ifndef CUDA_CUDAFUNCS_H__
#define CUDA_CUDAFUNCS_H__

#include <stdlib.h>
#include <cuda_runtime_api.h>
#include "Types.h"

/* FIXME: undef somewhere. */
#ifdef _DEBUG
#define DEBUG_SYNC {throwOnError(cudaGetLastError()); throwOnError(cudaDeviceSynchronize()); }
#else
#define DEBUG_SYNC
#endif


namespace cuda_runtime {

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


#endif
