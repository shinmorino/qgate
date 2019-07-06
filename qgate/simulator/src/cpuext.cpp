#include "pyglue.h"
#include "CPUQubitStates.h"
#include "CPUQubitProcessor.h"
#include "CPUQubitsStatesGetter.h"

namespace qcpu = qgate_cpu;

namespace {

extern "C"
PyObject *qubit_states_new(PyObject *module, PyObject *args) {
    
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "O", &dtype))
        return NULL;

    qgate::QubitStates *qstates = NULL;
    if (isFloat64(dtype))
        qstates = new qcpu::CPUQubitStates<double>();
    else if (isFloat32(dtype))
        qstates = new qcpu::CPUQubitStates<float>();
    else
        assert("Must not reach.");
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)qstates);
    return obj;
}


extern "C"
PyObject *qubit_processor_new(PyObject *module, PyObject *args) {
    
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "O", &dtype))
        return NULL;

    /* FIXME; add type specifier. */
    qgate::QubitProcessor *proc = NULL;
    if (isFloat64(dtype))
        proc = new qcpu::CPUQubitProcessor<double>();
    else if (isFloat32(dtype))
        proc = new qcpu::CPUQubitProcessor<float>();
    else
        assert("Must not reach.");
    
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)proc);
    return obj;
}

extern "C"
PyObject *qubits_states_getter_new(PyObject *module, PyObject *args) {
    
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "O", &dtype))
        return NULL;

    /* FIXME; add type specifier. */
    qgate::QubitsStatesGetter *proc = NULL;
    if (isFloat64(dtype))
        proc = new qcpu::CPUQubitsStatesGetter<double>();
    else if (isFloat32(dtype))
        proc = new qcpu::CPUQubitsStatesGetter<float>();
    else
        assert("Must not reach.");
    
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)proc);
    return obj;
}


static
PyMethodDef cpuext_methods[] = {
    {"qubit_states_new", qubit_states_new, METH_VARARGS},
    {"qubit_processor_new", qubit_processor_new, METH_VARARGS},
    {"qubits_states_getter_new", qubits_states_getter_new, METH_VARARGS},
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
        cpuext_methods,
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
    PyObject *module = Py_InitModule(modname, cpuext_methods);
    if (module == NULL)
        return;
    import_array();
}

#endif
