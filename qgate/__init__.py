from . import model
from . import simulator
from .model import prefs

def dump(obj, file = None, number_format = None) :
    import sys
    if file is None :
        file = sys.stdout
    
    from .simulator import qubits
    from .simulator import value_store
    from .simulator import utils
    
    if isinstance(obj, (model.GateList, list)) :
        model.gatelist.dump(obj, file)
    elif isinstance(obj, (qubits.Qubits, qubits.StateGetter)) :
        utils.dump_qubits(obj, file, number_format)
    elif isinstance(obj, value_store.ValueStore) :
        utils.dump_values(obj, file)
    else :
        raise RuntimeError('unknow object, {}.'.format(type(obj)))
