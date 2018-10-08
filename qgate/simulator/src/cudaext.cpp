/* -*- c++ -*- */

#include "pyglue.h"
#include "CUDARuntime.h"

namespace {

using namespace cuda_runtime;


CUDARuntime *cudaRuntime(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<CUDARuntime*>(val);
}

CUDAQubits *cudaQubits(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<CUDAQubits*>(val);
}


extern "C"
PyObject *runtime_new(PyObject *module, PyObject *args) {
    CUDARuntime *runtime = new CUDARuntime();
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)runtime);
    return obj;
}

extern "C"
PyObject *runtime_delete(PyObject *module, PyObject *args) {
    PyObject *objExt;
    if (!PyArg_ParseTuple(args, "O", &objExt))
        return NULL;
    
    npy_uint64 val = PyArrayScalar_VAL(objExt, UInt64);
    CUDARuntime *runtime = reinterpret_cast<CUDARuntime*>(val);
    delete runtime;

    Py_INCREF(Py_None);
    return Py_None;
}

extern "C"
PyObject *runtime_set_qubits(PyObject *module, PyObject *args) {
    PyObject *objExt, *objQubits;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &objQubits))
        return NULL;

    npy_uint64 val;
    val = PyArrayScalar_VAL(objExt, UInt64);
    CUDARuntime *runtime = reinterpret_cast<CUDARuntime*>(val);
    val = PyArrayScalar_VAL(objQubits, UInt64);
    CUDAQubits *qubits = reinterpret_cast<CUDAQubits*>(val);
    runtime->setQubits(qubits);

    Py_INCREF(Py_None);
    return Py_None;
}

extern "C"
PyObject *runtime_set_all_qreg_ids(PyObject *module, PyObject *args) {
    PyObject *objExt, *objQregList;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &objQregList))
        return NULL;
    
    IdList qregIdList = toIdList(objQregList);
    cudaRuntime(objExt)->setAllQregIds(qregIdList);
    
    Py_INCREF(Py_None);
    return Py_None;
}

extern "C"
PyObject *runtime_allocate_qubit_states(PyObject *module, PyObject *args) {
    PyObject *objExt, *objQregIdList;
    int circuitIdx;
    if (!PyArg_ParseTuple(args, "OiO", &objExt, &circuitIdx, &objQregIdList))
        return NULL;
    
    IdList qregIdList = toIdList(objQregIdList);
    cudaRuntime(objExt)->allocateQubitStates(circuitIdx, qregIdList);
    
    Py_INCREF(Py_None);
    return Py_None;
}


extern "C"
PyObject *runtime_get_qubits(PyObject *module, PyObject *args) {
    PyObject *objExt;
    if (!PyArg_ParseTuple(args, "O", &objExt))
        return NULL;

    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)&cudaRuntime(objExt)->getQubits());
    return obj;
}


extern "C"
PyObject *runtime_measure(PyObject *module, PyObject *args) {
    PyObject *objExt;
    int circuitIdx, qregId;
    double randNum;
    if (!PyArg_ParseTuple(args, "Odii", &objExt, &randNum, &circuitIdx, &qregId))
        return NULL;

    int cregVal = cudaRuntime(objExt)->measure((real)randNum, circuitIdx, qregId);
    return Py_BuildValue("i", cregVal);
}


extern "C"
PyObject *runtime_apply_reset(PyObject *module, PyObject *args) {
    PyObject *objExt;
    int circuitIdx, qregId;
    if (!PyArg_ParseTuple(args, "Oii", &objExt, &circuitIdx, &qregId))
        return NULL;

    cudaRuntime(objExt)->applyReset(circuitIdx, qregId);
    
    Py_INCREF(Py_None);
    return Py_None;
}


extern "C"
PyObject *runtime_apply_unary_gate(PyObject *module, PyObject *args) {
    PyObject *objExt, *objMat2x2;
    int circuitIdx, qregId;
    if (!PyArg_ParseTuple(args, "OOii", &objExt, &objMat2x2, &circuitIdx, &qregId))
        return NULL;

    CMatrix2x2 mat;
    matrix2x2FromNdArray(mat, objMat2x2);
    cudaRuntime(objExt)->applyUnaryGate(mat, circuitIdx, qregId);
    
    Py_INCREF(Py_None);
    return Py_None;
}


extern "C"
PyObject *runtime_apply_control_gate(PyObject *module, PyObject *args) {
    PyObject *objExt, *objMat2x2;
    int circuitIdx, controlId, targetId;
    if (!PyArg_ParseTuple(args, "OOiii", &objExt, &objMat2x2, &circuitIdx, &controlId, &targetId))
        return NULL;

    CMatrix2x2 mat;
    matrix2x2FromNdArray(mat, objMat2x2);
    cudaRuntime(objExt)->applyControlGate(mat, circuitIdx, controlId, targetId);
    
    Py_INCREF(Py_None);
    return Py_None;
}


extern "C"
PyObject *qubits_new(PyObject *module, PyObject *args) {
    CUDAQubits *runtime = new CUDAQubits();
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
    CUDAQubits *qubits = reinterpret_cast<CUDAQubits*>(val);
    delete qubits;

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
    cudaQubits(objExt)->getStates(&probBuf[arrayOffset], beginIdx, endIdx);
    
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
    cudaQubits(objExt)->getProbabilities(&probBuf[arrayOffset], beginIdx, endIdx);
    
    Py_INCREF(Py_None);
    return Py_None;
}


static
PyMethodDef formulas_methods[] = {
    {"runtime_new", runtime_new, METH_VARARGS},
    {"runtime_delete", runtime_delete, METH_VARARGS},
    {"runtime_set_qubits", runtime_set_qubits, METH_VARARGS},
    {"runtime_set_all_qreg_ids", runtime_set_all_qreg_ids, METH_VARARGS},
    {"runtime_allocate_qubit_states", runtime_allocate_qubit_states, METH_VARARGS},
    {"runtime_get_qubits", runtime_get_qubits, METH_VARARGS},
    {"runtime_measure", runtime_measure, METH_VARARGS},
    {"runtime_apply_reset", runtime_apply_reset, METH_VARARGS},
    {"runtime_apply_unary_gate", runtime_apply_unary_gate, METH_VARARGS},
    {"runtime_apply_control_gate", runtime_apply_control_gate, METH_VARARGS},
    {"qubits_new", qubits_new, METH_VARARGS},
    {"qubits_delete", qubits_delete, METH_VARARGS},
    {"qubits_get_states", qubits_get_states, METH_VARARGS},
    {"qubits_get_probabilities", qubits_get_probabilities, METH_VARARGS},
    {NULL},
};

}



#define modname "cudaext"
#define INIT_MODULE INITFUNCNAME(cudaext)

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
