import qgate.model as model
    
def new_circuit() :
    return model.Clause()

def allocate_qreg() :
    return model.Qreg()

def allocate_qregs(count) :
    return [model.Qreg() for _ in range(count)]

def allocate_creg() :
    return model.Creg()

def allocate_cregs(count) :
    return [model.Creg() for _ in range(count)]

# functions to instantiate operators

def measure(qregs, cregs) :
    return model.Measure(qregs, cregs)

def barrier(*qregs) :
    bar = model.Barrier(qregs)
    return bar

def reset(*qregs) :
    reset = model.Reset(qregs)
    return reset
        
def clause(*ops) :
    cl = model.Clause()
    for op in ops :
        cl.add_op(op)
    return cl

def if_(cregs, val, ops) :
    if isinstance(cregs, model.Creg) :
        cregs = [cregs]
    if_clause = model.IfClause(cregs, val)
    cl = clause(ops)
    if_clause.set_clause(cl)
    return if_clause
