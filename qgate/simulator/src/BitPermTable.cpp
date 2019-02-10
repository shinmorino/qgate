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
    
    qgate::IdList bitShiftMap(size);
    if (size != 0) {
        IdList sorted = bitPosList;
        std::sort(sorted.begin(), sorted.end());
        for (int idx = 0; idx < (int)sorted.size(); ++idx) {
            int inBitPos = sorted[idx] - idx;
            assert(inBitPos < size + bitPosList.size());
            if (inBitPos < size)
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
