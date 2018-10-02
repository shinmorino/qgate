
def _ket_format(n_qubits) :
    # {0:0'd'b}
    return '|{{0:0{0:d}b}}> '.format(n_qubits)

_number_format = '{1:.3f}'


def _dump_array(array, n_qubits, number_format) :
    format = _ket_format(n_qubits) + number_format
    for idx, value in enumerate(array) :
        print(format.format(idx, value))

def dump_qubit_states(qubits, number_format = _number_format) :
    _dump_array(qubits.get_states(), qubits.get_n_qubits(), number_format)

def dump_probabilities(qubits, number_format = _number_format) :
    _dump_array(qubits.get_probabilities(), qubits.get_n_qubits(), number_format)

def dump_creg_values(creg_dict) :
    for creg_array in creg_dict.get_arrays() :
        print(creg_array)
        values = creg_dict.get_values(creg_array)
        for idx, value in enumerate(values) :
            print("{:d}:".format(idx), value)

