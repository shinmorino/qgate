#pragma once

namespace qgate_cuda {

class MultiDeviceMemoryStore;
class CUDADevices;

extern CUDADevices cudaDevices;
extern MultiDeviceMemoryStore cudaMemoryStore;

}
