/* -*- c++ -*- */
#pragma once

#if defined(_WIN32) && defined(_DEBUG)
#  undef _DEBUG
#  include <Python.h>
#  define _DEBUG
#else
#  include <Python.h>
#endif
#include <bytesobject.h>


#if PY_MAJOR_VERSION >= 3

#define IsIntegerType(o) (PyLong_Check(o))

#else

#define IsIntegerType(o) (PyLong_Check(o) || PyInt_Check(o))

#endif


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <stdlib.h>
#include <algorithm>

#define throwError(a)
#define throwErrorIf(a, b) if (a) abort();


#include "Types.h"

namespace {

template<class T> inline
void throwErrorForInvalidArray(PyObject *obj) {
    throwError("Unsupported type.");
}

template<> inline
void throwErrorForInvalidArray<float>(PyObject *obj) {
    bool ok = PyArray_Check(obj) && (PyArray_TYPE((PyArrayObject*)obj) == NPY_FLOAT);
    throwErrorIf(!(ok), "Invalid array type.");
}

template<> inline
void throwErrorForInvalidArray<std::complex<float>>(PyObject *obj) {
    bool ok = PyArray_Check(obj) && (PyArray_TYPE((PyArrayObject*)obj) == NPY_CFLOAT);
    throwErrorIf(!(ok), "Invalid array type.");
}

template<> inline
void throwErrorForInvalidArray<double>(PyObject *obj) {
    bool ok = PyArray_Check(obj) && (PyArray_TYPE((PyArrayObject*)obj) == NPY_DOUBLE);
    throwErrorIf(!(ok), "Invalid array type.");
}

template<> inline
void throwErrorForInvalidArray<char>(PyObject *obj) {
    bool ok = PyArray_Check(obj) && (PyArray_TYPE((PyArrayObject*)obj) == NPY_INT8);
    throwErrorIf(!(ok), "Invalid array type.");
}


inline
PyObject *newScalarObj(double v) {
    PyObject *obj = PyArrayScalar_New(Float64);
    PyArrayScalar_ASSIGN(obj, Float64, (npy_float64)v);
    return obj;
}

inline
PyObject *newScalarObj(float v) {
    PyObject *obj = PyArrayScalar_New(Float32);
    PyArrayScalar_ASSIGN(obj, Float32, (npy_float32)v);
    return obj;
}



/* Helpers for dtypes */
inline
bool isFloat64(PyObject *dtype) {
    /* Since PyFloat64ArrType_Type may be defined as another type, using PyDoubleArrType_Type. */
    return dtype == (PyObject*)&PyDoubleArrType_Type;
}
inline
bool isFloat32(PyObject *dtype) {
    /* Since PyFloat32ArrType_Type may be defined as another type, using PyFloatArrType_Type. */
    return dtype == (PyObject*)&PyFloatArrType_Type;
}


IdList toIdList(PyObject *pyObj) {
    PyObject *iter = PyObject_GetIter(pyObj);
    PyObject *item;

    IdList idList;
    
    while ((item = PyIter_Next(iter)) != NULL) {
        int v = PyLong_AsLong(item);
        Py_DECREF(item);
        idList.push_back(v);
    }
    Py_DECREF(iter);
    
    return idList;
}

inline
void matrix2x2FromNdArray(CMatrix2x2 &mat, PyObject *pyObj) {
    throwErrorForInvalidArray<std::complex<real>>(pyObj);
    PyArrayObject *arr = (PyArrayObject*)pyObj;
    Complex *data = (Complex*)PyArray_DATA(arr);
    assert(PyArray_NDIM(arr) == 2);
    int stride = (int)PyArray_STRIDE(arr, 0) / sizeof(Complex);
    mat(0, 0) = data[0];
    mat(0, 1) = data[1];
    mat(1, 0) = data[stride];
    mat(1, 1) = data[stride + 1];
}


real *getArrayBuffer(PyObject *pyObj, QstateIdxType *size) {
    throwErrorForInvalidArray<real>(pyObj);
    PyArrayObject *arr = (PyArrayObject*)pyObj;
    real *data = (real*)PyArray_DATA(arr);
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


} /* anonymous namespace */

/* exception handling macro */

#define TRY try
#define CATCH_ERROR_AND_RETURN                      \
        catch (const std::exception &e) {                   \
            PyErr_SetString(PyExc_RuntimeError, e.what());  \
            return NULL;                                    \
        }



#if PY_MAJOR_VERSION >= 3

#define INITFUNCNAME(name) PyInit_##name

#else

#define INITFUNCNAME(name) init##name

#endif



/* references 
 * http://owa.as.wakwak.ne.jp/zope/docs/Python/BindingC/
 * http://scipy-cookbook.readthedocs.io/items/C_Extensions_NumPy_arrays.html
 *
 * Input validation is minimal in C++ side, assuming required checkes are done in python.
 */
