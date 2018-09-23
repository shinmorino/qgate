import sys
this = sys.modules[__name__]

from .model import measure, barrier, reset, clause, if_c
from . import model


def allocate_qreg(count) :
    return this.program.allocate_qreg(count)
    
def allocate_creg(count) :
    return this.program.allocate_creg(count)


this.program = None
    
def new_program() :
    this.program = model.Program()

def fin_program() :
    this.program = None
#    model.Operator.remove_regfunc(_add_operator)
    
def current_program() :
    return this.program

def op(*ops) :
    for op in ops :
        this.program.add_op(op)

