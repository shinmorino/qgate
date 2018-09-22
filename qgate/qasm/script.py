import sys
this = sys.modules[__name__]

from . import model


def allocate_qreg(count) :
    return this.program.allocate_qreg(count)
    
def allocate_creg(count) :
    return this.program.allocate_creg(count)


def _add_operator(op) :
    this.program.add_op(op)

def init_program() :
    this.program = model.Program()
    model.Operator.add_regfunc(_add_operator)

def fin_program() :
    this.program = None
    model.Operator.remove_regfunc(_add_operator)
    
def current_program() :
    return this.program

def measure(qregs, cregs) :
    measure = model.Measure(qregs, cregs)

def barrier(*qregs) :
    bar = model.Barrier(qregs)
    this.program.add_op(measure)
 
def if_c(creg, val, ops) :
    pass
