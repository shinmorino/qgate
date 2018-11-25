/* -*- c++ -*- */

#include "pyglue.h"
#include "CUDAQubitStates.h"
#include "CUDAQubitProcessor.h"
#include "CUDADevice.h"

namespace qcuda = qgate_cuda;

namespace {

const char *rsrc_key = "cuda_device";

qcuda::CUDADevice *cudaDevice(PyObject *module) {
    PyObject *objDict = PyModule_GetDict(module);
    PyObject *objDevice = PyDict_GetItemString(objDict, rsrc_key);
    npy_uint64 val = PyArrayScalar_VAL(objDevice, UInt64);
    return reinterpret_cast<qcuda::CUDADevice*>(val);
}

void module_init(PyObject *module) {
    qcuda::CUDADevice *dev = new qcuda::CUDADevice();
    try {
        dev->initialize(0);
    }
    catch (...) {
        delete dev;
        throw;
    }
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)dev);

    PyModule_AddObject(module, rsrc_key, obj);
}

PyObject *module_finalize(PyObject *module, PyObject *) {
    qcuda::CUDADevice *device = cudaDevice(module);
    device->finalize();
    delete device;
    cudaDeviceReset();
    Py_INCREF(Py_None);
    return Py_None;
}


extern "C"
PyObject *qubit_states_new(PyObject *module, PyObject *args) {
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "O", &dtype))
        return NULL;
    
    qcuda::CUDADevice *device = cudaDevice(module);
    qgate::QubitStates *qstates = NULL;
    if (isFloat64(dtype))
        qstates = new qcuda::CUDAQubitStates<double>(device);
    else if (isFloat32(dtype))
        qstates = new qcuda::CUDAQubitStates<float>(device);
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

    qcuda::CUDADevice *dev = cudaDevice(module);
    
    qgate::QubitProcessor *qproc = NULL;
    if (isFloat64(dtype))
        qproc = new qcuda::CUDAQubitProcessor<double>(*dev);
    else if (isFloat32(dtype))
        qproc = new qcuda::CUDAQubitProcessor<float>(*dev);
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
