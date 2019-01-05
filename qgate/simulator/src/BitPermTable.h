#pragma once

#include "Types.h"
#include <algorithm>
#include <string.h>

namespace qgate {

struct BitPermTable {

    /* create bit permulation table */
    void init_QregIdxToLocalIdx(const IdList &qregIdList) {
        memset(tables_, 0, sizeof(tables_));
        int maxQregId = *std::max_element(qregIdList.begin(), qregIdList.end());
        nTables_ = divru(maxQregId + 1, 8);
        int nLanes = (int)qregIdList.size();
        for (int laneIdx = 0; laneIdx < nLanes; ++laneIdx) {
            int qregIdx = qregIdList[laneIdx];
            int qregTableIdx = qregIdx / 8;
            QstateIdx qregBit = Qone << (qregIdx % 8);
            QstateIdx laneBit = Qone << laneIdx;
            for (int idx = 0; idx < 256; ++idx) {
                if (idx & qregBit)
                    tables_[qregTableIdx][idx] |= laneBit;
            }
        }
    }

    template<class F>
    void init_idxToQstateIdx(int nBitsInIdx, const F &permuteFunc) {
        memset(tables_, 0, sizeof(tables_));
        nTables_ = divru(nBitsInIdx, 8);
        for (int iBit = 0; iBit < nBitsInIdx; ++iBit) {
            QstateIdx permuted = permuteFunc(Qone << iBit);
            int tableIdx = iBit / 8;
            QstateIdx idxBit = Qone << (iBit % 8);
            for (int idx = 0; idx < 256; ++idx) {
                if (idx & idxBit)
                    tables_[tableIdx][idx] |= permuted;
            }
        }
    }
    
    QstateIdx permute(QstateIdx idx) const {
        unsigned char *ucharIdx = reinterpret_cast<unsigned char *>(&idx);
        QstateIdx permuted = 0;
        for (int iTable = 0; iTable < nTables_; ++iTable)
            permuted |= tables_[iTable][ucharIdx[iTable]];
        return permuted;
    }
    
    QstateIdx permute_56bits(QstateIdx idx) const {
        QstateIdx hiBits = 0;
        unsigned char *ucharIdx = reinterpret_cast<unsigned char *>(&idx);
        for (int iTable = 1; iTable < nTables_; ++iTable)
            hiBits |= tables_[iTable][ucharIdx[iTable]];
        return hiBits;
    }

    QstateIdx permute_8bits(QstateIdx hiBits, int idx) const {
        return hiBits | tables_[0][idx];
    }
    
    QstateIdx tables_[8][256];
    int nTables_;
};

}

