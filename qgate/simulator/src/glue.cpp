#include "pyglue.h"
#include "Interfaces.h"
#include "GateMatrix.h"
#include "Misc.h"

namespace {

typedef void (*MatFunc_0)(qgate::Matrix2x2C64 &mat);
typedef void (*MatFunc_1)(qgate::Matrix2x2C64 &mat, double);
typedef void (*MatFunc_2)(qgate::Matrix2x2C64 &mat, double, double);
typedef void (*MatFunc_3)(qgate::Matrix2x2C64 &mat, double, double, double);

double getTupleItemValue(PyObject *tuple, int idx) {
    PyObject *item = PyTuple_GET_ITEM(tuple, idx);
    return PyFloat_AsDouble(item);
}

struct MatFactory {
    virtual void operator()(qgate::Matrix2x2C64 *mat, PyObject *args) const = 0;
};


/* gate matrix */
struct MatFactory_0 : MatFactory {
    MatFactory_0(const MatFunc_0 f) : matfunc_(f) { }
    void operator()(qgate::Matrix2x2C64 *mat, PyObject *args) const {
        matfunc_(*mat);
    }
    const MatFunc_0 matfunc_;
};

struct MatFactory_1 : MatFactory {
    MatFactory_1(const MatFunc_1 f) : matfunc_(f) { }
    void operator()(qgate::Matrix2x2C64 *mat, PyObject *args)const  {
        double v0 = getTupleItemValue(args, 0);
        matfunc_(*mat, v0);
    }
    const MatFunc_1 matfunc_;
};

struct MatFactory_2 : MatFactory {
    MatFactory_2(const MatFunc_2 f) : matfunc_(f) { }
    void operator()(qgate::Matrix2x2C64 *mat, PyObject *args) const {
        double v0 = getTupleItemValue(args, 0);
        double v1 = getTupleItemValue(args, 1);
        matfunc_(*mat, v0, v1);
    }
    const MatFunc_2 matfunc_;
};

struct MatFactory_3 : MatFactory {
    MatFactory_3(const MatFunc_3 f) : matfunc_(f) { }
    void operator()(qgate::Matrix2x2C64 *mat, PyObject *args) const {
        double v0 = getTupleItemValue(args, 0);
        double v1 = getTupleItemValue(args, 1);
        double v2 = getTupleItemValue(args, 2);
        matfunc_(*mat, v0, v1, v2);
    }
    const MatFunc_3 matfunc_;
};

MatFactory_3 gen_U(qgate::U_mat);
MatFactory_2 gen_U2(qgate::U2_mat);
MatFactory_1 gen_U1(qgate::U1_mat);

MatFactory_0 gen_ID(qgate::ID_mat);
MatFactory_0 gen_X(qgate::X_mat);
MatFactory_0 gen_Y(qgate::Y_mat);
MatFactory_0 gen_Z(qgate::Z_mat);
MatFactory_0 gen_H(qgate::H_mat);
MatFactory_0 gen_S(qgate::S_mat);
MatFactory_0 gen_T(qgate::T_mat);

MatFactory_1 gen_RX(qgate::RX_mat);
MatFactory_1 gen_RY(qgate::RY_mat);
MatFactory_1 gen_RZ(qgate::RY_mat);

MatFactory_1 gen_ExpiI(qgate::ExpiI_mat);
MatFactory_1 gen_ExpiZ(qgate::ExpiZ_mat);

MatFactory_0 gen_SH(qgate::SH_mat);


PyObject *genPtrObj(const MatFactory &matFactory) {
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)&matFactory);
    return obj;
}


/* registration : Gate matrix factory */
extern "C"
PyObject *register_matrix_factory(PyObject *module, PyObject *args) {
    PyObject *objGateType;
    if (!PyArg_ParseTuple(args, "O", &objGateType))
        return NULL;

    const char *gateType = getStringFromObject(objGateType);
    if (strcmp(gateType, "U") == 0)
        return genPtrObj(gen_U);
    if (strcmp(gateType, "U2") == 0)
        return genPtrObj(gen_U2);
    if (strcmp(gateType, "U1") == 0)
        return genPtrObj(gen_U1);
    if (strcmp(gateType, "ID") == 0)
        return genPtrObj(gen_ID);
    if (strcmp(gateType, "X") == 0)
        return genPtrObj(gen_X);
    if (strcmp(gateType, "Y") == 0)
        return genPtrObj(gen_Y);
    if (strcmp(gateType, "Z") == 0)
        return genPtrObj(gen_Z);
    if (strcmp(gateType, "H") == 0)
        return genPtrObj(gen_H);
    if (strcmp(gateType, "S") == 0)
        return genPtrObj(gen_S);
    if (strcmp(gateType, "T") == 0)
        return genPtrObj(gen_T);
    if (strcmp(gateType, "RX") == 0)
        return genPtrObj(gen_RX);
    if (strcmp(gateType, "RY") == 0)
        return genPtrObj(gen_RY);
    if (strcmp(gateType, "RZ") == 0)
        return genPtrObj(gen_RZ);
    if (strcmp(gateType, "ExpiI") == 0)
        return genPtrObj(gen_ExpiI);
    if (strcmp(gateType, "ExpiZ") == 0)
        return genPtrObj(gen_ExpiZ);
    if (strcmp(gateType, "SH") == 0)
        return genPtrObj(gen_SH);

    PyErr_SetString(PyExc_RuntimeError, "Unknown gate type.");
    return NULL;
}


void *getArrayBuffer(PyObject *pyObj, qgate::QstateIdx *size) {
    PyArrayObject *arr = (PyArrayObject*)pyObj;
    void *data = PyArray_DATA(arr);
    throwErrorIf(3 <= PyArray_NDIM(arr), "ndarray is not 1-diemsional.");
    if (PyArray_NDIM(arr) == 2) {
        int rows = (int)PyArray_SHAPE(arr)[0];
        int cols = (int)PyArray_SHAPE(arr)[1];
        throwErrorIf((rows != 1) && (cols != 1), "ndarray is not 1-diemsional.");
        *size = std::max(rows, cols);
    }
    else /*if (PyArray_NDIM(arr) == 1) */  {
        *size = (int)PyArray_SHAPE(arr)[0];
    }
    return data;
}


qgate::QubitStates *qubitStates(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<qgate::QubitStates*>(val);
}

qgate::QubitProcessor *qproc(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<qgate::QubitProcessor*>(val);
}

qgate::SamplingPool *samplingPool(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<qgate::SamplingPool*>(val);
}

const MatFactory &matFactory(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return *reinterpret_cast<MatFactory*>(val);
}

extern "C"
PyObject *qubit_states_deallocate(PyObject *module, PyObject *args) {
    PyObject *objQstates;
    if (!PyArg_ParseTuple(args, "O", &objQstates))
        return NULL;

    qubitStates(objQstates)->deallocate();

    Py_INCREF(Py_None);
    return Py_None;
}


extern "C"
PyObject *qubit_states_delete(PyObject *module, PyObject *args) {
    PyObject *objExt;
    if (!PyArg_ParseTuple(args, "O", &objExt))
        return NULL;
    
    npy_uint64 val = PyArrayScalar_VAL(objExt, UInt64);
    qgate::QubitStates *qstates = reinterpret_cast<qgate::QubitStates*>(val);
    delete qstates;

    Py_INCREF(Py_None);
    return Py_None;
}


extern "C"
PyObject *qubit_processor_delete(PyObject *module, PyObject *args) {
    PyObject *objExt;
    if (!PyArg_ParseTuple(args, "O", &objExt))
        return NULL;
    
    npy_uint64 val = PyArrayScalar_VAL(objExt, UInt64);
    qgate::QubitProcessor *qproc = reinterpret_cast<qgate::QubitProcessor*>(val);
    delete qproc;

    Py_INCREF(Py_None);
    return Py_None;
}


extern "C"
PyObject *qubit_states_get_n_lanes(PyObject *module, PyObject *args) {
    PyObject *objQstates;
    if (!PyArg_ParseTuple(args, "O", &objQstates))
        return NULL;

    int nQregs = qubitStates(objQstates)->getNLanes();

    return Py_BuildValue("i", nQregs);
}


/* qubit processor */

extern "C"
PyObject *qubit_processor_reset(PyObject *module, PyObject *args) {
    PyObject *objQproc;
    if (!PyArg_ParseTuple(args, "O", &objQproc))
        return NULL;

    qproc(objQproc)->reset();

    Py_INCREF(Py_None);
    return Py_None;
}

extern "C"
PyObject *qubit_processor_initialize_qubit_states(PyObject *module, PyObject *args) {
    PyObject *objQproc, *objQstates;
    int nLanes;
    if (!PyArg_ParseTuple(args, "OOi", &objQproc, &objQstates, &nLanes))
        return NULL;

    qgate::QubitStates *qstates = qubitStates(objQstates);
    qproc(objQproc)->initializeQubitStates(*qstates, nLanes);
    
    Py_INCREF(Py_None);
    return Py_None;
}

extern "C"
PyObject *qubit_processor_reset_qubit_states(PyObject *module, PyObject *args) {
    PyObject *objQproc, *objQstates;
    if (!PyArg_ParseTuple(args, "OO", &objQproc, &objQstates))
        return NULL;

    qgate::QubitStates *qstates = qubitStates(objQstates);
    qproc(objQproc)->resetQubitStates(*qstates);
    
    Py_INCREF(Py_None);
    return Py_None;
}

extern "C"
PyObject *qubit_processor_calc_probability(PyObject *module, PyObject *args) {
    PyObject *objQproc, *objQstates;
    int qregId;
    if (!PyArg_ParseTuple(args, "OOi", &objQproc, &objQstates, &qregId))
        return NULL;

    qgate::QubitStates *qstates = qubitStates(objQstates);
    double prob = qproc(objQproc)->calcProbability(*qstates, qregId);
    return Py_BuildValue("d", prob);
}

extern "C"
PyObject *qubit_processor_join(PyObject *module, PyObject *args) {
    PyObject *objQproc, *objQstates, *objQstatesList;
    int nNewLanes;
    if (!PyArg_ParseTuple(args, "OOOi", &objQproc, &objQstates, &objQstatesList, &nNewLanes))
        return NULL;

    qgate::QubitStates *qstates = qubitStates(objQstates);
    qgate::QubitStatesList qstatesList;
    PyObject *iter = PyObject_GetIter(objQstatesList);
    PyObject *item;
    while ((item = PyIter_Next(iter)) != NULL) {
        qgate::QubitStates *qstates = qubitStates(item);
        Py_DECREF(item);
        qstatesList.push_back(qstates);
    }
    Py_DECREF(iter);
    
    qproc(objQproc)->join(*qstates, qstatesList, nNewLanes);
    Py_INCREF(Py_None);
    return Py_None;
}

extern "C"
PyObject *qubit_processor_decohere(PyObject *module, PyObject *args) {
    PyObject *objQproc, *objQstates;
    int value, localLane;
    double prob;
    if (!PyArg_ParseTuple(args, "OidOi", &objQproc, &value, &prob, &objQstates, &localLane))
        return NULL;

    qgate::QubitStates *qstates = qubitStates(objQstates);
    qproc(objQproc)->decohere(value, prob, *qstates, localLane);
    Py_INCREF(Py_None);
    return Py_None;
}

extern "C"
PyObject *qubit_processor_decohere_and_separate(PyObject *module, PyObject *args) {
    PyObject *objQproc, *objQstates0, *objQstates1, *objQstates;
    int value, localLane;
    double prob;
    if (!PyArg_ParseTuple(args, "OidOOOi", &objQproc, &value, &prob,
                          &objQstates0, &objQstates1, &objQstates, &localLane))
        return NULL;

    qgate::QubitStates *qstates0 = qubitStates(objQstates0);
    qgate::QubitStates *qstates1 = qubitStates(objQstates1);
    qgate::QubitStates *qstates = qubitStates(objQstates);
    qproc(objQproc)->decohereAndSeparate(value, prob, *qstates0, *qstates1, *qstates, localLane);
    Py_INCREF(Py_None);
    return Py_None;
}

extern "C"
PyObject *qubit_processor_apply_reset(PyObject *module, PyObject *args) {
    PyObject *objQproc, *objQstates;
    int localLane;
    if (!PyArg_ParseTuple(args, "OOi", &objQproc, &objQstates, &localLane))
        return NULL;

    qgate::QubitStates *qstates = qubitStates(objQstates);
    qproc(objQproc)->applyReset(*qstates, localLane);
    
    Py_INCREF(Py_None);
    return Py_None;
}


extern "C"
PyObject *qubit_processor_apply_unary_gate(PyObject *module, PyObject *args) {
    PyObject *objQproc, *objGateType, *objAdjoint, *objQstates;
    int localLane;
    if (!PyArg_ParseTuple(args, "OOOOi", &objQproc, &objGateType, &objAdjoint,
                          &objQstates, &localLane))
        return NULL;

    PyObject *objCmatf = PyObject_GetAttrString(objGateType, "cmatf");
    assert(objCmatf != NULL);
    const MatFactory &factory = matFactory(objCmatf);
    PyObject *objGateArgs = PyObject_GetAttrString(objGateType, "args");

    qgate::Matrix2x2C64 mat;
    factory(&mat, objGateArgs);
    if (objAdjoint == Py_True)
        adjoint(&mat);
    
    qgate::QubitStates *qstates = qubitStates(objQstates);
    qproc(objQproc)->applyUnaryGate(mat, *qstates, localLane);
    
    Py_INCREF(Py_None);
    return Py_None;
}


extern "C"
PyObject *qubit_processor_apply_control_gate(PyObject *module, PyObject *args) {
    PyObject *objQproc, *objGateType, *objAdjoint, *objQstates, *objLocalControlLanes;
    int localTargetLane;
    if (!PyArg_ParseTuple(args, "OOOOOi", &objQproc, &objGateType, &objAdjoint,
                          &objQstates, &objLocalControlLanes, &localTargetLane))
        return NULL;

    PyObject *objCmatf = PyObject_GetAttrString(objGateType, "cmatf");
    const MatFactory &factory = matFactory(objCmatf);
    PyObject *objGateArgs = PyObject_GetAttrString(objGateType, "args");

    qgate::Matrix2x2C64 mat;
    factory(&mat, objGateArgs);
    if (objAdjoint == Py_True)
        adjoint(&mat);
    
    qgate::QubitStates *qstates = qubitStates(objQstates);
    qgate::IdList localControlLanes = toIdList(objLocalControlLanes);
    qproc(objQproc)->applyControlGate(mat, *qstates, localControlLanes, localTargetLane);
    
    Py_INCREF(Py_None);
    return Py_None;
}


qgate::QubitStatesList toQubitStatesList(PyObject *objQstatesList) {
    qgate::QubitStatesList qstatesList;
    PyObject *iter = PyObject_GetIter(objQstatesList);
    PyObject *item;
    while ((item = PyIter_Next(iter)) != NULL) {
        qgate::QubitStates *qstates = qubitStates(item);
        Py_DECREF(item);
        qstatesList.push_back(qstates);
    }
    Py_DECREF(iter);
    return qstatesList;
}

extern "C"
PyObject *qubit_processor_get_states(PyObject *module, PyObject *args) {
    PyObject *objQproc, *objArray, *objLocalToExt, *objQstatesList;
    int mathOp;
    qgate::QstateIdx arrayOffset, start, step;
    qgate::QstateSize nStates, emptyLaneMask, nExtLanes;
    
    if (!PyArg_ParseTuple(args, "OOKiOKOKKKK",
                          &objQproc,
                          &objArray, &arrayOffset,
                          &mathOp,
                          &objLocalToExt, &emptyLaneMask, &objQstatesList, &nExtLanes,
                          &nStates, &start, &step)) {
        return NULL;
    }

    qgate::QstateIdx arraySize = 0;
    void *array = getArrayBuffer(objArray, &arraySize);
    if (arraySize < nStates) {
        PyErr_SetString(PyExc_ValueError, "array size too small.");
        return NULL;
    }
    qgate::QstateSize nQstatesSize = qgate::Qone << nExtLanes;
    if ((start < 0) || (nQstatesSize <= start)) {
        PyErr_SetString(PyExc_ValueError, "value out of range");
        return NULL;
    }
    qgate::QstateIdx end = start + step * (nStates - 1);
    if ((end < 0) || (nQstatesSize <= end)) {
        PyErr_SetString(PyExc_ValueError, "value out of range");
        return NULL;
    }

    qgate::QubitStatesList qstatesList = toQubitStatesList(objQstatesList);
    if (qstatesList.empty()) {
        qgate::QstateSize itemSize = PyArray_ITEMSIZE((PyArrayObject*)objArray);
        qgate::QstateSize byteSize = nStates * itemSize;
        qgate::fillZeros(array, byteSize);
        if (start == 0) {
            /* this code works for both prob(scalar) and state(complex). */
            if (mathOp == qgate::mathOpNull)
                itemSize /= 2;
            if (itemSize == 4)
                *static_cast<float*>(array) = 1.f;
            else
                *static_cast<double*>(array) = 1.;
        }
        Py_INCREF(Py_None);
        return Py_None;
    }

    /* ext to local */
    qgate::IdListList localToExt = toIdListList(objLocalToExt);
    qproc(objQproc)->getStates(array, arrayOffset, (qgate::MathOp)mathOp,
                               localToExt.data(), emptyLaneMask, qstatesList, nStates, start, step);

    Py_INCREF(Py_None);
    return Py_None;
}

extern "C"
PyObject *qubit_processor_prepare_prob_array(PyObject *module, PyObject *args) {
    PyObject *objQproc, *objArray, *objLocalToExt, *objQstatesList;
    qgate::QstateSize nLanes, nHiddenLanes;

    if (!PyArg_ParseTuple(args, "OOOOii",
                          &objQproc, &objArray,
                          &objLocalToExt, &objQstatesList,
                          &nLanes, &nHiddenLanes)) {
        return NULL;
    }

    qgate::QstateSize nStates = qgate::Qone << nLanes;
    qgate::QstateIdx arraySize = 0;
    void *array = getArrayBuffer(objArray, &arraySize);
    if (arraySize < nStates) {
        PyErr_SetString(PyExc_ValueError, "array size too small.");
        return NULL;
    }

    /* ext to local */
    qgate::QubitStatesList qstatesList = toQubitStatesList(objQstatesList);
    qgate::IdListList localToExt = toIdListList(objLocalToExt);
    qproc(objQproc)->prepareProbArray(array,
                                      localToExt, qstatesList, nLanes, nHiddenLanes);

    Py_INCREF(Py_None);
    return Py_None;
}

extern "C"
PyObject *qubit_processor_create_sampling_pool(PyObject *module, PyObject *args) {
    PyObject *objQproc, *objLocalToExt, *objQstatesList, *objEmptyLanes;
    qgate::QstateSize nLanes, nHiddenLanes;

    if (!PyArg_ParseTuple(args, "OOOiiO",
                          &objQproc,
                          &objLocalToExt, &objQstatesList, &nLanes, &nHiddenLanes, &objEmptyLanes)) {
        return NULL;
    }

    qgate::QubitStatesList qstatesList = toQubitStatesList(objQstatesList);
    qgate::IdListList localToExt = toIdListList(objLocalToExt);
    qgate::IdList emptyLanes = toIdList(objEmptyLanes);
    qgate::SamplingPool *spool = qproc(objQproc)->createSamplingPool(localToExt, qstatesList, nLanes, nHiddenLanes, emptyLanes);

    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)spool);
    return obj;
}

extern "C"
PyObject *sampling_pool_sample(PyObject *module, PyObject *args) {
    PyObject *objSamplingPool, *objObs, *objRandNum;
    int nSamples;

    if (!PyArg_ParseTuple(args, "OOiO",
                          &objSamplingPool, &objObs, &nSamples, &objRandNum)) {
        return NULL;
    }

    npy_intp itemSize = PyArray_ITEMSIZE((PyArrayObject*)objObs);
    assert(itemSize == sizeof(qgate::QstateIdx));

    qgate::QstateIdx arraySize = 0;
    void *obsArray = getArrayBuffer(objObs, &arraySize);
    if (arraySize < nSamples) {
        PyErr_SetString(PyExc_ValueError, "array size too small.");
        return NULL;
    }
    arraySize = 0;
    void *randNum = getArrayBuffer(objRandNum, &arraySize);
    if (arraySize < nSamples) {
        PyErr_SetString(PyExc_ValueError, "array size too small.");
        return NULL;
    }
    samplingPool(objSamplingPool)->sample(static_cast<qgate::QstateIdx*>(obsArray), nSamples, static_cast<double*>(randNum));
    Py_INCREF(Py_None);
    return Py_None;
}

extern "C"
PyObject *sampling_pool_delete(PyObject *module, PyObject *args) {
    PyObject *objSamplingPool;

    if (!PyArg_ParseTuple(args, "O", &objSamplingPool)) {
        return NULL;
    }
    delete samplingPool(objSamplingPool);

    Py_INCREF(Py_None);
    return Py_None;
}


static
PyMethodDef glue_methods[] = {
    {"register_matrix_factory", register_matrix_factory, METH_VARARGS},
    {"qubit_states_delete", qubit_states_delete, METH_VARARGS},
    {"qubit_processor_delete", qubit_processor_delete, METH_VARARGS},
    {"qubit_states_get_n_lanes", qubit_states_get_n_lanes, METH_VARARGS},
    {"qubit_states_deallocate", qubit_states_deallocate, METH_VARARGS },
    {"qubit_processor_reset", qubit_processor_reset, METH_VARARGS },
    {"qubit_processor_initialize_qubit_states", qubit_processor_initialize_qubit_states, METH_VARARGS},
    {"qubit_processor_reset_qubit_states", qubit_processor_reset_qubit_states, METH_VARARGS},
    {"qubit_processor_calc_probability", qubit_processor_calc_probability, METH_VARARGS},
    {"qubit_processor_join", qubit_processor_join, METH_VARARGS},
    {"qubit_processor_decohere", qubit_processor_decohere, METH_VARARGS},
    {"qubit_processor_decohere_and_separate", qubit_processor_decohere_and_separate, METH_VARARGS},
    {"qubit_processor_apply_reset", qubit_processor_apply_reset, METH_VARARGS},
    {"qubit_processor_apply_unary_gate", qubit_processor_apply_unary_gate, METH_VARARGS},
    {"qubit_processor_apply_control_gate", qubit_processor_apply_control_gate, METH_VARARGS},
    {"qubit_processor_get_states", qubit_processor_get_states, METH_VARARGS},
    {"qubit_processor_prepare_prob_array", qubit_processor_prepare_prob_array, METH_VARARGS},
    {"qubit_processor_create_sampling_pool", qubit_processor_create_sampling_pool, METH_VARARGS},
    {"sampling_pool_sample", sampling_pool_sample, METH_VARARGS},
    {"sampling_pool_delete", sampling_pool_delete, METH_VARARGS},

    {NULL},
};

}


#define modname "glue"
#define INIT_MODULE INITFUNCNAME(glue)

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        modname,
        NULL, 0,
        glue_methods,
        NULL, NULL, NULL, NULL
};

extern "C"
PyMODINIT_FUNC INIT_MODULE(void) {
    PyObject *module = PyModule_Create(&moduledef);
    if (module == NULL)
        return NULL;
    import_array();
    return module;
}

#else

PyMODINIT_FUNC INIT_MODULE(void) {
    PyObject *module = Py_InitModule(modname, glue_methods);
    if (module == NULL)
        return;
    import_array();
}

#endif
