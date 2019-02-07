#include "BitPermTable.h"

using namespace qgate;

IdList qgate::createBitShiftMap(const int bitPos, int size) {
    qgate::IdList bitShiftMap(size);
    for (int idx = 0; idx < (int)bitShiftMap.size(); ++idx) {
        bitShiftMap[idx] = idx;
        if (bitPos <= idx)
            bitShiftMap[idx] += 1;
    }
    return bitShiftMap;
}

IdList qgate::createBitShiftMap(const IdList &bitPosList, int size) {
    
#if 0
    /* FIXME: std::vector<>::resize() did not work on g++. */
    qgate::IdList bitShiftMap(size);
#else
    qgate::IdList bitShiftMap;
    for (int idx = 0; idx < size; ++idx)
        bitShiftMap.push_back(0);
#endif
    assert(idxToQstateIdx.size() == size);
    if (size != 0) {
        IdList sorted = bitPosList;
        std::sort(sorted.begin(), sorted.end());
        for (int idx = 0; idx < (int)sorted.size(); ++idx) {
            int inBitPos = sorted[idx] - idx;
            assert(inBitPos < size);
            bitShiftMap[inBitPos] += 1;
        }
        for (int idx = 0; idx < size - 1; ++idx)
            bitShiftMap[idx + 1] += bitShiftMap[idx];
        for (int idx = 0; idx < size; ++idx)
            bitShiftMap[idx] += idx;
    }
    return bitShiftMap;
}

QstateIdx qgate::createBitmask(const IdList &bitPosList) {
    QstateIdx mask = 0;
    for (int idx = 0; idx < (int)bitPosList.size(); ++idx)
        mask |= Qone << bitPosList[idx];
    return mask;
}
