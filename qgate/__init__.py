from . import model
from .simulator import utils
from .model import prefs

def dump(obj, mathop = None, number_format = None) :
    if isinstance(obj, (model.GateList, list)) :
        model.gatelist.dump(obj)
    else :
        utils.dump(obj, mathop, number_format)
