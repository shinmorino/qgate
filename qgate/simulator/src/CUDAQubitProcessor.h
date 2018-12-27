#pragma once

#include "Interfaces.h"
#include "DeviceTypes.h"
#include "CUDADevice.h"
#include <map>

namespace qgate_cuda {

using qgate::Matrix2x2C64;
using qgate::QubitStates;
using qgate::MathOp;
using qgate::QstateIdx;
using qgate::QstateSize;
using qgate::QubitStatesList;

/* forwarded decls. */
template<class R> class CUDAQubitStates;
template<class R> class DeviceProcPrimitives;


template<class real>
class CUDAQubitProcessor : public qgate::QubitProcessor {
    typedef qgate::ComplexType<real> Complex;
    typedef DeviceComplexType<real> DeviceComplex;
    typedef CUDAQubitStates<real> CUQStates;
public:
    CUDAQubitProcessor(CUDADevices &devices);
    ~CUDAQubitProcessor();

    virtual void clear();

    virtual void prepare();

    virtual void initializeQubitStates(const qgate::IdList &qregIdList, qgate::QubitStates &qstates,
                                       int nLanesPerDevice, qgate::IdList &_deviceIds);

    virtual void resetQubitStates(qgate::QubitStates &qstates);
    
    virtual int measure(double randNum, QubitStates &qstates, int qregId);
    
    virtual void applyReset(QubitStates &qstates, int qregId);

    virtual void applyUnaryGate(const Matrix2x2C64 &mat, QubitStates &qstates, int qregId);

    virtual void applyControlGate(const Matrix2x2C64 &mat, QubitStates &qstates, int controlId, int targetId);

    virtual void getStates(void *array, QstateIdx arrayOffset,
                           MathOp op,
                           const QubitStatesList &qstatesList,
                           QstateIdx beginIdx, QstateIdx endIdx);
	
    /* synchronize all active devices */
    void synchronize();

    /* template methods to use device lambda.
     * They're intented for private use, though placed on public space to enable device lamgda. */

    template<class F>
    void dispatchToDevices(CUQStates &cuQstates, const F &f);

    template<class F>
    void dispatchToDevices(CUQStates &cuQstates, const F &f, QstateIdx begin, QstateIdx end, QstateSize nThreadsPerDevice);

    template<class F>
    void apply(const qgate::IdList &lanes, CUQStates &cuQstates, F &f);

    template<class F>
    void apply(int bitPos, CUQStates &cuQstates, const F &f);

    template<class F>
    void applyHi(int bitPos, CUQStates &cuQstates, const F &f);

    template<class F>
    void applyLo(int bitPos, CUQStates &cuQstates, const F &f);

    template<class F>
    void apply(int bitPos, CUQStates &cuQstates, const F &f,
               qgate::QstateSize nThreads, bool runHi, bool runLo);

private:

    CUDADevices &devices_;
    typedef std::map<int, DeviceProcPrimitives<real>*> ProcMap;
    ProcMap procMap_;
};
        
}
