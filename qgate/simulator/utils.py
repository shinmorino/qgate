from __future__ import print_function
from . import qubits
from . import value_store
import io

def _ket_format(n_qubits) :
    # {0:0'd'b}
    return '|{{0:0{0:d}b}}> '.format(n_qubits)

_complex_number_format = '{1: .3f}'
_prob_number_format = '{1:g}'


def dump_qubits(obj, file, number_format = None) :
    if isinstance(obj, qubits.Qubits) :
        obj = obj.states
    
    if number_format is None :
        if obj.mathop == qubits.null :
            number_format = _complex_number_format
        else :
            number_format = _prob_number_format

    n_qubits = obj.n_lanes
    array = obj[:]
    format = _ket_format(n_qubits) + number_format
    for idx, value in enumerate(array) :
        print(format.format(idx, value), file = file)

def dump_values(value_store, file) :
    for key, value in value_store.valuedict.items() :
        print("{:d}:".format(key), value, file = file)


