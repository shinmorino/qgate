from __future__ import print_function, unicode_literals
from . import qubits
from . import value_store
from .observation import Observation, ObservationList, ObservationHistgram
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

    n_qubits = obj.n_qregs
    array = obj[:]
    format = _ket_format(n_qubits) + number_format
    for idx, value in enumerate(array) :
        print(format.format(idx, value), file = file)

def dump_values(value_store, file) :
    for key, value in value_store.valuedict.items() :
        print("{:d}:".format(key), value, file = file)

def _observation_to_repr(dump_format, value, none_mask) :
    vrepr = dump_format.format(value)
    mrepr = dump_format.format(none_mask)
    obsrepr = [value_c if mask_c == '0' else '*' for value_c, mask_c in zip(vrepr, mrepr)]
    return ''.join(obsrepr)

def observation_repr(self) :
    dump_format = '{{:0{}b}}'.format(len(self._reflist))
    return _observation_to_repr(dump_format, self._value, self._none_mask)

Observation.__repr__ = observation_repr

def dump_oblist(obj, file) :
    values = obj._values
    mask = obj._mask
    dump_format = '{{:0{}b}}'.format(len(obj._reflist))
    for value in values :
        print(_observation_to_repr(dump_format, value, mask), file = file)

def observation_list_repr(self) :
    values = self._values
    mask = self._mask
    dump_format = '{{:0{}b}}'.format(len(self._reflist))
    obsreprlist = list()
    if len(self) <= 256 :
        for idx in range(len(self)) :
            obsrepr = _observation_to_repr(dump_format, values[idx], mask)
            obsreprlist.append(obsrepr)
        return '[' + ', '.join(obsreprlist) + ']'
    for idx in range(0, 3) :
        obsrepr = _observation_to_repr(dump_format, values[idx], mask)
        obsreprlist.append(obsrepr)
    n_obs = len(self)
    for idx in range(n_obs - 3, n_obs) :
        obsrepr = _observation_to_repr(dump_format, values[idx], mask)
        obsreprlist.append(obsrepr)
    return '[' + ', '.join(obsreprlist[0:3]) + ', ... , ' + ', '.join(obsreprlist[-3:]) + ']'

ObservationList.__repr__ = observation_list_repr

def dump_obshist(obs, file) :
    dump_format = '{{:0{}b}}: {{}}'.format(obs._n_bits)
    hist = obs._hist
    keys = sorted(hist.keys())
    for key in keys :
        print(dump_format.format(key, hist[key]), file = file)

def observation_histgram_repr(self) :
    hist = self._hist
    dump_format = '{{:0{}b}}: {{}}'.format(self._n_bits)
    if len(hist) <= 256 :
        keys = sorted(hist.keys())
        itemreprlist = [dump_format.format(key, hist[key]) for key in keys]
        return '{' + ', '.join(itemreprlist) + '}'
    
    keys = sorted(hist.keys())
    itemreprlist_0 = [dump_format.format(key, hist[key]) for key in keys[:3]]
    itemreprlist_1 = [dump_format.format(key, hist[key]) for key in keys[-3:]]
    return '{' + ', '.join(itemreprlist_0) + ', ... , ' + ', '.join(itemreprlist_1) + '}'

ObservationHistgram.__repr__ = observation_histgram_repr
