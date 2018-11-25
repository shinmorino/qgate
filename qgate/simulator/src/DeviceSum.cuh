#include "CUDADevice.h"

using namespace qgate_cuda;

#define FULL_MASK 0xffffffff

template<class real, class F>
__global__
void sumKernel(real *d_partialSum, qgate::QstateIdx offset, const F f, qgate::QstateIdx size) {
    qgate::QstateIdx gid = blockDim.x * blockIdx.x + threadIdx.x;
    qgate::QstateIdx stride = gridDim.x * blockDim.x;

    real sum = real();
    for (qgate::QstateIdx idx = gid; idx < size; idx += stride) {
        sum += f(idx + offset);
    }
    
    /* reduce in 128-thread blocks */
    sum += __shfl_xor_sync(FULL_MASK, sum, 16); 
    sum += __shfl_xor_sync(FULL_MASK, sum, 8); 
    sum += __shfl_xor_sync(FULL_MASK, sum, 4); 
    sum += __shfl_xor_sync(FULL_MASK, sum, 2); 
    sum += __shfl_xor_sync(FULL_MASK, sum, 1);

    int laneId = threadIdx.x % warpSize;
    int laneId4 = laneId % 4;
    int warpId = threadIdx.x / warpSize;
    __shared__ real partialSum[4];
    if (laneId == 0)
        partialSum[warpId] = sum;
    __syncthreads();
    if (warpId == 0) {
        sum = partialSum[laneId4];
        sum += __shfl_xor_sync(FULL_MASK, sum, 2); 
        sum += __shfl_xor_sync(FULL_MASK, sum, 1);
        if (laneId == 0)
            d_partialSum[blockIdx.x] = sum;
    }
}

template<class V, class F>
V deviceSum(CUDADevice &dev,
            qgate::QstateIdx begin, qgate::QstateIdx end, const F &f) {
    
    int nBlocks;
    throwOnError(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nBlocks, sumKernel<V, F>, 128, 0));
    V *h_partialSum = dev.getTmpHostMem<V>(nBlocks);
    
    sumKernel<<<nBlocks, 128>>>(h_partialSum, begin, f, end - begin);
    DEBUG_SYNC;
    throwOnError(cudaDeviceSynchronize()); /* FIXME: add stream. */
    V sum = V();
    for (int idx = 0; idx < nBlocks; ++idx)
        sum += h_partialSum[idx];
    return sum;
}
