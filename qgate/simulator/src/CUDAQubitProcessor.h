#pragma once

#include "Interfaces.h"
#include "DeviceTypes.h"
#include "CUDADevice.h"
#include "MultiChunkPtr.h"
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
    typedef MultiChunkPtr<DeviceComplex> DevicePtr;
public:
    CUDAQubitProcessor(CUDADevices &devices);
    ~CUDAQubitProcessor();

    virtual void clear();

    virtual void prepare();

    virtual void initializeQubitStates(const qgate::IdList &qregIdList, qgate::QubitStates &qstates,
                                       int nLanesPerChunk, qgate::IdList &_deviceIds);

    virtual void resetQubitStates(qgate::QubitStates &qstates);

    virtual double calcProbability(const qgate::QubitStates &qstates, int qregId);
    
    virtual int measure(double randNum, QubitStates &qstates, int qregId);
    
    virtual void applyReset(QubitStates &qstates, int qregId);

    virtual void applyUnaryGate(const Matrix2x2C64 &mat, QubitStates &qstates, int qregId);

    virtual void applyControlGate(const Matrix2x2C64 &mat, QubitStates &qstates, int controlId, int targetId);

    virtual void getStates(void *array, QstateIdx arrayOffset,
                           MathOp op,
                           const QubitStatesList &qstatesList,
                           QstateIdx beginIdx, QstateIdx endIdx, QstateIdx step);
	
    /* synchronize all active devices */
    void synchronize();

    /* synchronize when using multiple devices */
    void synchronizeMultiDevice();

    /* template methods to use device lambda.
     * They're intented for private use, though placed on public space to enable device lamgda. */

    template<class F>
    void dispatch(const qgate::IdList &lanes, CUQStates &cuQstates, F &f);

    template<class F>
    void dispatch(int bitPos, CUQStates &cuQstates, const F &f);

    template<class F>
    void dispatch(const qgate::IdList &ordered, const F &f, QstateIdx begin, QstateIdx end);

private:
    qgate::IdList orderChunks(int bitPos, const CUQStates &cuQstates, bool runHi, bool runLo) const;

    real _calcProbability(const CUDAQubitStates<real> &qstates, int lane);

    CUDADevices &devices_;
    typedef std::vector<DeviceProcPrimitives<real>*> Procs;
    Procs procs_;
    CUDADeviceList activeDevices_;
};
        
}
