/* -*- c++ -*- */

#include "pyglue.h"
#include "CPURuntime.h"

namespace {


CPUQubits *cpuQubits(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<CPUQubits*>(val);
}


CPUQubitStates *cpuQubitStates(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<CPUQubitStates*>(val);
}

extern "C"
PyObject *qubits_new(PyObject *module, PyObject *args) {
    CPUQubits *runtime = new CPUQubits();
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)runtime);
    return obj;
}

extern "C"
PyObject *qubits_delete(PyObject *module, PyObject *args) {
    PyObject *objExt;
    if (!PyArg_ParseTuple(args, "O", &objExt))
        return NULL;
    
    npy_uint64 val = PyArrayScalar_VAL(objExt, UInt64);
    CPUQubits *qubits = reinterpret_cast<CPUQubits*>(val);
    delete qubits;

    Py_INCREF(Py_None);
    return Py_None;
}

extern "C"
PyObject *qubits_add_qubit_states(PyObject *module, PyObject *args) {
    PyObject *objExt, *objQstates;
    int key;
    if (!PyArg_ParseTuple(args, "OiO", &objExt, &key, &objQstates))
        return NULL;
    
    CPUQubitStates *qstates = cpuQubitStates(objQstates);
    cpuQubits(objExt)->addQubitStates(key, qstates);
    
    Py_INCREF(Py_None);
    return Py_None;
}

extern "C"
PyObject *qubits_detach_qubit_states(PyObject *module, PyObject *args) {
    PyObject *objExt;
    if (!PyArg_ParseTuple(args, "O", &objExt))
        return NULL;
    
    cpuQubits(objExt)->detachQubitStates();
    
    Py_INCREF(Py_None);
    return Py_None;
}

extern "C"
PyObject *qubits_get_states(PyObject *module, PyObject *args) {
    PyObject *objExt, *objArray;
    QstateIdxType beginIdx, endIdx, arrayOffset;
    if (!PyArg_ParseTuple(args, "OOKKK", &objExt, &objArray, &beginIdx, &endIdx, &arrayOffset))
        return NULL;
    
    QstateIdxType arraySize = 0;
    Complex *probBuf = getArrayBuffer<Complex>(objArray, &arraySize);
    throwErrorIf(arraySize < arrayOffset, "array offset is larger than array size.");
    QstateIdxType dstSize = arraySize - arrayOffset;

    QstateIdxType copySize = endIdx - beginIdx;
    throwErrorIf(dstSize < copySize, "array size too small.");

    probBuf += arrayOffset;
    cpuQubits(objExt)->getStates(&probBuf[arrayOffset], beginIdx, endIdx);
    
    Py_INCREF(Py_None);
    return Py_None;
}

extern "C"
PyObject *qubits_get_probabilities(PyObject *module, PyObject *args) {
    PyObject *objExt, *objArray;
    QstateIdxType beginIdx, endIdx, arrayOffset;
    if (!PyArg_ParseTuple(args, "OOKKK", &objExt, &objArray, &beginIdx, &endIdx, &arrayOffset))
        return NULL;
    
    QstateIdxType arraySize = 0;
    real *probBuf = getArrayBuffer<real>(objArray, &arraySize);
    throwErrorIf(arraySize < arrayOffset, "array offset is larger than array size.");
    QstateIdxType dstSize = arraySize - arrayOffset;

    QstateIdxType copySize = endIdx - beginIdx;
    throwErrorIf(dstSize < copySize, "array size too small.");

    probBuf += arrayOffset;
    cpuQubits(objExt)->getProbabilities(&probBuf[arrayOffset], beginIdx, endIdx);
    
    Py_INCREF(Py_None);
    return Py_None;
}


extern "C"
PyObject *qubit_states_new(PyObject *module, PyObject *args) {
    
    PyObject *objQregList;
    if (!PyArg_ParseTuple(args, "O", &objQregList))
        return NULL;
    
    IdList qregIdList = toIdList(objQregList);

    CPUQubitStates *qstates = new CPUQubitStates();
    qstates->allocate(qregIdList);
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)qstates);
    return obj;
}

extern "C"
PyObject *qubit_states_delete(PyObject *module, PyObject *args) {
    PyObject *objExt;
    if (!PyArg_ParseTuple(args, "O", &objExt))
        return NULL;
    
    npy_uint64 val = PyArrayScalar_VAL(objExt, UInt64);
    CPUQubitStates *qstates = reinterpret_cast<CPUQubitStates*>(val);
    delete qstates;

    Py_INCREF(Py_None);
    return Py_None;
}

extern "C"
PyObject *qubit_states_get_n_qregs(PyObject *module, PyObject *args) {
    PyObject *objQstates;
    if (!PyArg_ParseTuple(args, "O", &objQstates))
        return NULL;

    CPUQubitStates *qstates = cpuQubitStates(objQstates);
    int nQregs = qstates->getNLanes();
    return Py_BuildValue("i", nQregs);
}


extern "C"
PyObject *measure(PyObject *module, PyObject *args) {
    double randNum;
    PyObject *objQstates;
    int qregId;
    if (!PyArg_ParseTuple(args, "dOi", &randNum, &objQstates, &qregId))
        return NULL;

    CPUQubitStates *qstates = cpuQubitStates(objQstates);
    int cregVal = cpuMeasure((real)randNum, *qstates, qregId);
    return Py_BuildValue("i", cregVal);
}


extern "C"
PyObject *apply_reset(PyObject *module, PyObject *args) {
    PyObject *objQstates;
    int qregId;
    if (!PyArg_ParseTuple(args, "Oi", &objQstates, &qregId))
        return NULL;

    CPUQubitStates *qstates = cpuQubitStates(objQstates);
    cpuApplyReset(*qstates, qregId);
    
    Py_INCREF(Py_None);
    return Py_None;
}


extern "C"
PyObject *apply_unary_gate(PyObject *module, PyObject *args) {
    PyObject *objMat2x2, *objQstates;
    int qregId;
    if (!PyArg_ParseTuple(args, "OOi", &objMat2x2, &objQstates, &qregId))
        return NULL;

    CMatrix2x2 mat;
    matrix2x2FromNdArray(mat, objMat2x2);
    CPUQubitStates *qstates = cpuQubitStates(objQstates);
    cpuApplyUnaryGate(mat, *qstates, qregId);
    
    Py_INCREF(Py_None);
    return Py_None;
}


extern "C"
PyObject *apply_control_gate(PyObject *module, PyObject *args) {
    PyObject *objMat2x2, *objQstates;
    int controlId, targetId;
    if (!PyArg_ParseTuple(args, "OOii", &objMat2x2, &objQstates, &controlId, &targetId))
        return NULL;

    CMatrix2x2 mat;
    matrix2x2FromNdArray(mat, objMat2x2);
    CPUQubitStates *qstates = cpuQubitStates(objQstates);
    cpuApplyControlGate(mat, *qstates, controlId, targetId);
    
    Py_INCREF(Py_None);
    return Py_None;
}



static
PyMethodDef formulas_methods[] = {
    {"qubits_new", qubits_new, METH_VARARGS},
    {"qubits_delete", qubits_delete, METH_VARARGS},
    {"qubits_add_qubit_states", qubits_add_qubit_states, METH_VARARGS},
    {"qubits_detach_qubit_states", qubits_detach_qubit_states, METH_VARARGS},
    {"qubits_get_states", qubits_get_states, METH_VARARGS},
    {"qubits_get_probabilities", qubits_get_probabilities, METH_VARARGS},
    {"qubit_states_new", qubit_states_new, METH_VARARGS},
    {"qubit_states_delete", qubit_states_delete, METH_VARARGS},
    {"qubit_states_get_n_qregs", qubit_states_get_n_qregs, METH_VARARGS},
    {"measure", measure, METH_VARARGS},
    {"apply_reset", apply_reset, METH_VARARGS},
    {"apply_unary_gate", apply_unary_gate, METH_VARARGS},
    {"apply_control_gate", apply_control_gate, METH_VARARGS},
    {NULL},
};

}



#define modname "cpuext"
#define INIT_MODULE INITFUNCNAME(cpuext)

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        modname,
        NULL, 0,
        formulas_methods,
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
    PyObject *module = Py_InitModule(modname, formulas_methods);
    if (module == NULL)
        return;
    import_array();
}

#endif
