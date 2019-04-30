from __future__ import print_function
from . import qubits
from . import value_store

def _ket_format(n_qubits) :
    # {0:0'd'b}
    return '|{{0:0{0:d}b}}> '.format(n_qubits)

_number_format = '{1:.3f}'


def _dump_array(array, n_qubits, number_format) :
    format = _ket_format(n_qubits) + number_format
    for idx, value in enumerate(array) :
        print(format.format(idx, value))

def _dump_values(value_store) :
    for key, value in value_store.valuedict.items() :
        print("{:d}:".format(key), value)

def dump(obj, mathop = None, number_format = None) :
    if mathop is None :
        mathop = qubits.null
    if number_format is None :
        number_format = _number_format

    if isinstance(obj, qubits.Qubits) :
        _dump_array(obj.get_states(mathop), obj.get_n_lanes(), number_format)
    elif isinstance(obj, value_store.ValueStore) :
        _dump_values(obj)
    else :
        print('Unknown object, {}'.format(repr(obj)))
