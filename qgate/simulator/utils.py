from __future__ import print_function
from . import qubits

def _ket_format(n_qubits) :
    # {0:0'd'b}
    return '|{{0:0{0:d}b}}> '.format(n_qubits)

_number_format = '{1:.3f}'


def _dump_array(array, n_qubits, number_format) :
    format = _ket_format(n_qubits) + number_format
    for idx, value in enumerate(array) :
        print(format.format(idx, value))

def dump(qubits, mathop = qubits.null, number_format = _number_format) :
    _dump_array(qubits.get_states(mathop), qubits.get_n_lanes(), number_format)

# FIXME: merge to dump().
def dump_creg_values(creg_values) :
    for key, value in creg_values.cregdict.items() :
        print("{:d}:".format(key), value)
