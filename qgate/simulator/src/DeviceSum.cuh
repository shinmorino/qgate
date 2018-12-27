#include "CUDADevice.h"

using namespace qgate_cuda;

#define FULL_MASK 0xffffffff

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

template<class V>
struct DeviceSum {
    DeviceSum(CUDADevice &dev) : dev_(dev) {
        nBlocks_ = 0;
    }
    
    template<class F>
    void launch(qgate::QstateIdx begin, qgate::QstateIdx end, const F &f) {
        throwOnError(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nBlocks_,
                                                                   sumKernel<V, F>, 128, 0));
        h_partialSum_ = dev_.getTmpHostMem<V>(nBlocks_);
        /* FIXME: adjust nBlocks_ when (end - begin) is small. */
        sumKernel<<<nBlocks_, 128>>>(h_partialSum_, begin, f, end - begin);
        DEBUG_SYNC;
    }

    V sync() {
        throwOnError(cudaDeviceSynchronize()); /* FIXME: add stream. */
        V sum = V();
        for (int idx = 0; idx < nBlocks_; ++idx)
            sum += h_partialSum_[idx];
        return sum;
    }
    
    CUDADevice &dev_;
    V *h_partialSum_;
    int nBlocks_;
};

template<class V, class F>
V deviceSum(CUDADevice &dev,
            qgate::QstateIdx begin, qgate::QstateIdx end, const F &f) {
    
    DeviceSum<V> devSum(dev);
    devSum.launch(begin, end, f);
    return devSum.sync();
}
