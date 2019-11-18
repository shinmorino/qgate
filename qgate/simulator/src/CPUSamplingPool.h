#pragma once

#include "Interfaces.h"
#include "BitPermTable.h"

namespace qgate_cpu {

template<class real>
class CPUSamplingPool : public qgate::SamplingPool {
public:
    CPUSamplingPool(real *prob, int nLanes, const qgate::IdList &emptyLanes);

    virtual ~CPUSamplingPool();

    virtual void sample(qgate::QstateIdx *observations, int nSamples, const double *randNum);

private:
    qgate::BitPermTable perm_;
    real *cumprob_;
    qgate::QstateSize nStates_;
};

}
