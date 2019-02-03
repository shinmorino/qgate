#include "pyglue.h"
#include "Interfaces.h"

namespace {

qgate::IdList toIdList(PyObject *pyObj) {
    PyObject *iter = PyObject_GetIter(pyObj);
    PyObject *item;

    qgate::IdList idList;
    
    while ((item = PyIter_Next(iter)) != NULL) {
        int v = PyLong_AsLong(item);
        Py_DECREF(item);
        idList.push_back(v);
    }
    Py_DECREF(iter);
    
    return idList;
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
PyObject *qubit_processor_clear(PyObject *module, PyObject *args) {
    PyObject *objQproc;
    if (!PyArg_ParseTuple(args, "O", &objQproc))
        return NULL;

    qproc(objQproc)->clear();

    Py_INCREF(Py_None);
    return Py_None;
}

extern "C"
PyObject *qubit_processor_initialize_qubit_states(PyObject *module, PyObject *args) {
    PyObject *objQproc, *objQstates, *objDeviceIds;
    int nLanes, nLanesPerDevice;
    if (!PyArg_ParseTuple(args, "OOiiO", &objQproc, &objQstates,
                          &nLanes, &nLanesPerDevice, &objDeviceIds))
        return NULL;

    qgate::IdList deviceIds = toIdList(objDeviceIds);
    qgate::QubitStates *qstates = qubitStates(objQstates);
    qproc(objQproc)->initializeQubitStates(*qstates, nLanes, nLanesPerDevice, deviceIds);
    
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
PyObject *qubit_processor_prepare(PyObject *module, PyObject *args) {
    PyObject *objQproc;
    if (!PyArg_ParseTuple(args, "O", &objQproc))
        return NULL;

    qproc(objQproc)->prepare();

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
PyObject *qubit_processor_measure(PyObject *module, PyObject *args) {
    double randNum;
    PyObject *objQproc, *objQstates;
    int localLane;
    if (!PyArg_ParseTuple(args, "OdOi", &objQproc, &randNum, &objQstates, &localLane))
        return NULL;

    qgate::QubitStates *qstates = qubitStates(objQstates);
    int cregVal = qproc(objQproc)->measure(randNum, *qstates, localLane);
    return Py_BuildValue("i", cregVal);
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
    PyObject *objQproc, *objMat2x2, *objQstates;
    int localLane;
    if (!PyArg_ParseTuple(args, "OOOi", &objQproc, &objMat2x2, &objQstates, &localLane))
        return NULL;

    qgate::Matrix2x2C64 mat;
    matrix2x2FromNdArray(mat, objMat2x2);
    qgate::QubitStates *qstates = qubitStates(objQstates);
    qproc(objQproc)->applyUnaryGate(mat, *qstates, localLane);
    
    Py_INCREF(Py_None);
    return Py_None;
}


extern "C"
PyObject *qubit_processor_apply_control_gate(PyObject *module, PyObject *args) {
    PyObject *objMat2x2, *objQstates, *objQproc;
    int localControlLane, localTargetLane;
    if (!PyArg_ParseTuple(args, "OOOii", &objQproc, &objMat2x2,
                          &objQstates, &localControlLane, &localTargetLane))
        return NULL;

    qgate::Matrix2x2C64 mat;
    matrix2x2FromNdArray(mat, objMat2x2);
    qgate::QubitStates *qstates = qubitStates(objQstates);
    qproc(objQproc)->applyControlGate(mat, *qstates, localControlLane, localTargetLane);
    
    Py_INCREF(Py_None);
    return Py_None;
}


extern "C"
PyObject *qubit_processor_get_states(PyObject *module, PyObject *args) {
    PyObject *objQproc, *objArray, *objLocalToExt, *objQstatesList;
    int mathOp;
    qgate::QstateIdx arrayOffset, start, step;
    qgate::QstateSize nStates, nExtLanes;
    
    if (!PyArg_ParseTuple(args, "OOKiOOKKKK",
                          &objQproc,
                          &objArray, &arrayOffset,
                          &mathOp,
                          &objLocalToExt, &objQstatesList, &nExtLanes,
                          &nStates, &start, &step)) {
        return NULL;
    }

    qgate::QstateIdx arraySize = 0;
    void *array = getArrayBuffer(objArray, &arraySize);

    qgate::QubitStatesList qstatesList;

    PyObject *iter = PyObject_GetIter(objQstatesList);
    PyObject *item;
    while ((item = PyIter_Next(iter)) != NULL) {
        qgate::QubitStates *qstates = qubitStates(item);
        Py_DECREF(item);
        qstatesList.push_back(qstates);
    }
    Py_DECREF(iter);
    
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

    /* ext to local */
    std::vector<qgate::IdList> localToExt;
    iter = PyObject_GetIter(objLocalToExt);
    while ((item = PyIter_Next(iter)) != NULL) {
        qgate::IdList ids = toIdList(item);
        Py_DECREF(item);
        localToExt.push_back(ids);
    }
    Py_DECREF(iter);
    
    qproc(objQproc)->getStates(array, arrayOffset, (qgate::MathOp)mathOp,
                               localToExt.data(), qstatesList, nStates, start, step);

    Py_INCREF(Py_None);
    return Py_None;
}


static
PyMethodDef glue_methods[] = {
    {"qubit_states_delete", qubit_states_delete, METH_VARARGS},
    {"qubit_processor_delete", qubit_processor_delete, METH_VARARGS},
    {"qubit_states_get_n_lanes", qubit_states_get_n_lanes, METH_VARARGS},
    {"qubit_states_deallocate", qubit_states_deallocate, METH_VARARGS },
    {"qubit_processor_clear", qubit_processor_clear, METH_VARARGS },
    {"qubit_processor_prepare", qubit_processor_prepare, METH_VARARGS },
    {"qubit_processor_initialize_qubit_states", qubit_processor_initialize_qubit_states, METH_VARARGS},
    {"qubit_processor_reset_qubit_states", qubit_processor_reset_qubit_states, METH_VARARGS},
    {"qubit_processor_calc_probability", qubit_processor_calc_probability, METH_VARARGS},
    {"qubit_processor_measure", qubit_processor_measure, METH_VARARGS},
    {"qubit_processor_apply_reset", qubit_processor_apply_reset, METH_VARARGS},
    {"qubit_processor_apply_unary_gate", qubit_processor_apply_unary_gate, METH_VARARGS},
    {"qubit_processor_apply_control_gate", qubit_processor_apply_control_gate, METH_VARARGS},
    {"qubit_processor_get_states", qubit_processor_get_states, METH_VARARGS},
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
