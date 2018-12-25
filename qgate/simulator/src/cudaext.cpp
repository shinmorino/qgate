/* -*- c++ -*- */

#include "pyglue.h"
#include "CUDAQubitStates.h"
#include "CUDAQubitProcessor.h"
#include "CUDADevice.h"

namespace qcuda = qgate_cuda;

namespace {

const char *rsrc_key = "cuda_devices";

qcuda::CUDADevices *cudaDevices(PyObject *module) {
    PyObject *objDict = PyModule_GetDict(module);
    PyObject *objDevice = PyDict_GetItemString(objDict, rsrc_key);
    npy_uint64 val = PyArrayScalar_VAL(objDevice, UInt64);
    return reinterpret_cast<qcuda::CUDADevices*>(val);
}

void module_init(PyObject *module) {
    qcuda::CUDADevices *devices = new qcuda::CUDADevices();
    qcuda::CUDADevice *device = NULL;
    try {
        devices->probe();
        device = devices->defaultDevice();
    }
    catch (...) {
        delete devices;
        throw;
    }
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)device);

    PyModule_AddObject(module, rsrc_key, obj);
}

PyObject *module_finalize(PyObject *module, PyObject *) {
    qcuda::CUDADevices *devices = cudaDevices(module);
    devices->finalize();
    delete devices;
    cudaDeviceReset();
    Py_INCREF(Py_None);
    return Py_None;
}


extern "C"
PyObject *qubit_states_new(PyObject *module, PyObject *args) {
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "O", &dtype))
        return NULL;
    
    qgate::QubitStates *qstates = NULL;
    if (isFloat64(dtype))
        qstates = new qcuda::CUDAQubitStates<double>();
    else if (isFloat32(dtype))
        qstates = new qcuda::CUDAQubitStates<float>();
    else
        abort_("unexpected dtype.");
    
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)qstates);
    return obj;
}


extern "C"
PyObject *qubit_processor_new(PyObject *module, PyObject *args) {
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "O", &dtype))
        return NULL;

    qcuda::CUDADevices *devices = cudaDevices(module);
    
    qgate::QubitProcessor *qproc = NULL;
    if (isFloat64(dtype))
        qproc = new qcuda::CUDAQubitProcessor<double>(*devices);
    else if (isFloat32(dtype))
        qproc = new qcuda::CUDAQubitProcessor<float>(*devices);
    else
        abort_("unexpected dtype.");
    
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)qproc);
    return obj;
}

static
PyMethodDef cudaext_methods[] = {
    {"module_finalize", module_finalize, METH_VARARGS},
    {"qubit_states_new", qubit_states_new, METH_VARARGS},
    {"qubit_processor_new", qubit_processor_new, METH_VARARGS},
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
        cudaext_methods,
        NULL, NULL, NULL, NULL
};

extern "C"
PyMODINIT_FUNC INIT_MODULE(void) {
    PyObject *module = PyModule_Create(&moduledef);
    if (module == NULL)
        return NULL;
    import_array();
    try {
        module_init(module);
    }
    catch (const std::runtime_error &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        Py_DECREF(module);
        return NULL;
    }
    return module;
}

#else

PyMODINIT_FUNC INIT_MODULE(void) {
    PyObject *module = Py_InitModule(modname, cudaext_methods);
    if (module == NULL)
        return;
    import_array();
    try {
        module_init(module);
    }
    catch (const std::runtime_error &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        Py_DECREF(module);
    }
}

#endif
