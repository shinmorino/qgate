

def dump_qubit_states(states) :
    for idx, value in enumerate(states) :
        print('{0:08b}'.format(idx), value)


def dump_creg_values(creg_dict) :
    for creg_array in creg_dict.get_creg_arrays() :
        print(creg_array)
        values = creg_dict.get_values(creg_array)
        for idx, value in enumerate(values) :
            print("{:d}:".format(idx), value)

