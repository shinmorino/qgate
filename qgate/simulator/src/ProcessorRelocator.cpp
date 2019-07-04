#include "ProcessorRelocator.h"
#include <valarray>
#include <algorithm>
#include "CUDAQubitStates.h"

using namespace qgate;
using namespace qgate_cuda;

void ProcessorRelocator::setLanes(const IdList &hiLanes, int loLane, int varLane) {
    masks_.clear();
    setBitMask_ = 0;
    varBit_ = 0;
     
    for (int idx = 0; idx < (int)hiLanes.size(); ++idx)
        setBitMask_ |=  1 << hiLanes[idx];
    
    IdList lanes = hiLanes;
    if (loLane != -1)
        lanes.push_back(loLane);
    if (varLane != -1) {
        lanes.push_back(varLane);
        varBit_ = 1 << varLane;
    }

    if (lanes.empty())
        return;

    std::sort(lanes.begin(), lanes.end());
    
    int nLanes = (int)lanes.size();
    std::valarray<int> masks_ms(nLanes), masks_ls(nLanes);

    for (int idx = 0; idx < nLanes; ++idx) {
        int bit = 1 << lanes[idx];
        masks_ms[idx] = ~(bit * 2 - 1);
        masks_ls[idx] = bit - 1;
    }

    masks_.push_back(masks_ls[0]);
    for (int idx = 0; idx < nLanes - 1; ++idx) {
        int mask = masks_ls[idx + 1] & masks_ms[idx];
        masks_.push_back(mask);
    }
    masks_.push_back(masks_ms[nLanes - 1]);
}

void ProcessorRelocator::setLanes(int loLane, int varLane) {
    setLanes(IdList(), loLane, varLane);
}

IdList ProcessorRelocator::generateIdxList(int nBits) {
    IdList generated;

    if (masks_.empty()) {
        int size = 1 << nBits;
        for (int idx = 0; idx < size; ++idx)
            generated.push_back(idx);
        return generated;
    }

    int nIdxBits = (int)masks_.size() - 1;
    int size = 1 << (nBits - nIdxBits);
    for (int idx = 0; idx < size; ++idx) {
        int shifted = setBitMask_;
        for (int shift = 0; shift < (int)masks_.size(); ++shift) {
            shifted |= (idx << shift) & masks_[shift];
        }
        generated.push_back(shifted);
        if (varBit_ != 0)
            generated.push_back(shifted | varBit_);
    }
    return generated;
}

template<class real>
IdList qgate::relocateProcessors(const CUDAQubitStates<real> &cuQStates, const IdList &_hiLanes, int loLane, int varLane) {
    int nLanes = cuQStates.getNLanes();
    int nLanesPerChunk = cuQStates.getNLanesPerChunk();
    int po2idx = nLanes - nLanesPerChunk;

    IdList hiLanes;
    for (auto &lane : _hiLanes) {
        int lanePrefix = lane - nLanesPerChunk;
        if (0 <= lanePrefix)
            hiLanes.push_back(lanePrefix);
    }
    loLane -= nLanesPerChunk;
    if (loLane < 0)
        loLane = -1;
    varLane -= nLanesPerChunk;
    if (varLane < 0)
        varLane = -1;

    ProcessorRelocator relocator;
    relocator.setLanes(hiLanes, loLane, varLane);
    return relocator.generateIdxList(po2idx);
}

template<class real>
IdList qgate::relocateProcessors(const CUDAQubitStates<real> &cuQStates, int loLane, int varLane) {
    return relocateProcessors(cuQStates, IdList(), loLane, varLane);
}

template IdList qgate::relocateProcessors(const CUDAQubitStates<float> &cuQStates, const IdList &_hiLanes, int loLane, int varLane);
template IdList qgate::relocateProcessors(const CUDAQubitStates<double> &cuQStates, const IdList &_hiLanes, int loLane, int varLane);
template IdList qgate::relocateProcessors(const CUDAQubitStates<float> &cuQStates, int loLane, int varLane);
template IdList qgate::relocateProcessors(const CUDAQubitStates<double> &cuQStates, int loLane, int varLane);
