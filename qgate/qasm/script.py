import sys
this = sys.modules[__name__]

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

def measure(qregs, cregs) :
    return model.Measure(qregs, cregs)

def barrier(*qregs) :
    bar = model.Barrier(qregs)
    this.program.add_op(bar)
    return bar

def reset(*qregs) :
    reset = model.Reset(qregs)
    this.program.add_op(reset)
    return reset
 
def if_c(creg, val, ops) :
    if_clause = model.IfClause(creg, val)
    cl = model.clause(ops)
    if_clause.set(cl)
    return if_clause


def op(*ops) :
    for op in ops :
        this.program.add_op(op)

