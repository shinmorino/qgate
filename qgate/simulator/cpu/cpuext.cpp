/* -*- c++ -*- */

#include "pyglue.h"
#include "CPUKernel.h"

namespace {


CPUKernel *cpuKernel(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<CPUKernel*>(val);
}

Qubits *cpuQubits(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<Qubits*>(val);
}


extern "C"
PyObject *kernel_new(PyObject *module, PyObject *args) {
    CPUKernel *kernel = new CPUKernel();
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)kernel);
    return obj;
}

extern "C"
PyObject *kernel_delete(PyObject *module, PyObject *args) {
    PyObject *objExt;
    if (!PyArg_ParseTuple(args, "O", &objExt))
        return NULL;
    
    npy_uint64 val = PyArrayScalar_VAL(objExt, UInt64);
    CPUKernel *kernel = reinterpret_cast<CPUKernel*>(val);
    delete kernel;

    Py_INCREF(Py_None);
    return Py_None;
}

extern "C"
PyObject *kernel_set_all_qreg_ids(PyObject *module, PyObject *args) {
    PyObject *objExt, *objQregList;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &objQregList))
        return NULL;
    
    IdList qregIdList = toIdList(objQregList);
    cpuKernel(objExt)->setAllQregIds(qregIdList);
    
    Py_INCREF(Py_None);
    return Py_None;
}

extern "C"
PyObject *kernel_allocate_qubit_states(PyObject *module, PyObject *args) {
    PyObject *objExt, *objQregIdList;
    int circuitIdx;
    if (!PyArg_ParseTuple(args, "OiO", &objExt, &circuitIdx, &objQregIdList))
        return NULL;
    
    IdList qregIdList = toIdList(objQregIdList);
    cpuKernel(objExt)->allocateQubitStates(circuitIdx, qregIdList);
    
    Py_INCREF(Py_None);
    return Py_None;
}


extern "C"
PyObject *kernel_get_qubits(PyObject *module, PyObject *args) {
    PyObject *objExt;
    if (!PyArg_ParseTuple(args, "O", &objExt))
        return NULL;

    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)&cpuKernel(objExt)->getQubits());
    return obj;
}


extern "C"
PyObject *kernel_measure(PyObject *module, PyObject *args) {
    PyObject *objExt;
    int circuitIdx, qregId;
    double randNum;
    if (!PyArg_ParseTuple(args, "Odii", &objExt, &randNum, &circuitIdx, &qregId))
        return NULL;

    int cregVal = cpuKernel(objExt)->measure((real)randNum, circuitIdx, qregId);
    return Py_BuildValue("i", cregVal);
}


extern "C"
PyObject *kernel_apply_reset(PyObject *module, PyObject *args) {
    PyObject *objExt;
    int circuitIdx, qregId;
    if (!PyArg_ParseTuple(args, "Oii", &objExt, &circuitIdx, &qregId))
        return NULL;

    cpuKernel(objExt)->applyReset(circuitIdx, qregId);
    
    Py_INCREF(Py_None);
    return Py_None;
}


extern "C"
PyObject *kernel_apply_unary_gate(PyObject *module, PyObject *args) {
    PyObject *objExt, *objMat2x2;
    int circuitIdx, qregId;
    if (!PyArg_ParseTuple(args, "OOii", &objExt, &objMat2x2, &circuitIdx, &qregId))
        return NULL;

    CMatrix2x2 mat;
    matrix2x2FromNdArray(mat, objMat2x2);
    cpuKernel(objExt)->applyUnaryGate(mat, circuitIdx, qregId);
    
    Py_INCREF(Py_None);
    return Py_None;
}


extern "C"
PyObject *kernel_apply_control_gate(PyObject *module, PyObject *args) {
    PyObject *objExt, *objMat2x2;
    int circuitIdx, controlId, targetId;
    if (!PyArg_ParseTuple(args, "OOiii", &objExt, &objMat2x2, &circuitIdx, &controlId, &targetId))
        return NULL;

    CMatrix2x2 mat;
    matrix2x2FromNdArray(mat, objMat2x2);
    cpuKernel(objExt)->applyControlGate(mat, circuitIdx, controlId, targetId);
    
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
    real *probBuf = getArrayBuffer(objArray, &arraySize);
    throwErrorIf(arraySize < arrayOffset, "array offset is larger than array size.");
    QstateIdxType dstSize = arraySize - arrayOffset;

    QstateIdxType copySize = endIdx - beginIdx;
    throwErrorIf(dstSize < copySize, "array size too small.");

    probBuf += arrayOffset;
    cpuQubits(objExt)->getProbabilities(&probBuf[arrayOffset], beginIdx, endIdx);
    
    Py_INCREF(Py_None);
    return Py_None;
}



static
PyMethodDef formulas_methods[] = {
    {"kernel_new", kernel_new, METH_VARARGS},
    {"kernel_delete", kernel_delete, METH_VARARGS},
    {"kernel_set_all_qreg_ids", kernel_set_all_qreg_ids, METH_VARARGS},
    {"kernel_allocate_qubit_states", kernel_allocate_qubit_states, METH_VARARGS},
    {"kernel_get_qubits", kernel_get_qubits, METH_VARARGS},
    {"kernel_measure", kernel_measure, METH_VARARGS},
    {"kernel_apply_reset", kernel_apply_reset, METH_VARARGS},
    {"kernel_apply_unary_gate", kernel_apply_unary_gate, METH_VARARGS},
    {"kernel_apply_control_gate", kernel_apply_control_gate, METH_VARARGS},
    {"qubits_get_probabilities", qubits_get_probabilities, METH_VARARGS},
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
