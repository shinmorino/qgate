

def dump_probabilities(qubits) :
    probs = qubits.get_probabilities()
    for idx, value in enumerate(probs) :
        print('{0:08b}'.format(idx), value)


def dump_creg_values(creg_dict) :
    for creg_array in creg_dict.get_arrays() :
        print(creg_array)
        values = creg_dict.get_values(creg_array)
        for idx, value in enumerate(values) :
            print("{:d}:".format(idx), value)

