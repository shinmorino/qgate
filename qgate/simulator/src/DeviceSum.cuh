#include "DeviceSum.h"

#define FULL_MASK 0xffffffff

namespace qgate_cuda {

template<class V, class F>
__global__
void sumKernel(V *d_partialSum, qgate::QstateIdx offset, const F f, qgate::QstateIdx size) {
    qgate::QstateIdx gid = blockDim.x * blockIdx.x + threadIdx.x;
    qgate::QstateIdx stride = gridDim.x * blockDim.x;

    V sum = V();
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
    __shared__ V partialSum[4];
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

template<class V> template<class F>
void DeviceSum<V>::launch(qgate::QstateIdx begin, qgate::QstateIdx end, const F &f) {
    dev_.makeCurrent(); /* FIXME: add stream. */
    throwOnError(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nBlocks_,
                                                               sumKernel<V, F>, 128, 0));
    SimpleMemoryStore hostMemStore = dev_.tempHostMemory();
    h_partialSum_ = hostMemStore.allocate<V>(nBlocks_);
    /* FIXME: adjust nBlocks_ when (end - begin) is small. */
    sumKernel<<<nBlocks_, 128>>>(h_partialSum_, begin, f, end - begin);
    DEBUG_SYNC;
}

}

