from . import model

class GateList :
    
    def __init__(self) :
        self.ops = []

    @staticmethod
    def _copy_list(ops) :
        copied = []
        for op in ops :
            if isinstance(op, model.Operator) :
                copied.append(op.copy())
            elif isinstance(op, (list, tuple)) :
                inner_gatelist = GateList()
                inner_gatelist.ops = GateList._copy_list(op)
                copied.append(inner_gatelist)
            else :
                assert False, 'Unknown argument, {}'.format(repr(op))
        return copied

    def __getitem__(self, key) :
        return self.ops[key]

    def __iter__(self) :
        return iter(self.ops)

    def copy(self) :
        copied = GateList()
        copied.ops = GateList._copy_list(self.ops)
        return copied

    def set(self, ops) :
        if isinstance(ops, model.Operator) :
            self.ops = [ops.copy()]
        elif isinstance(ops, GateList) :
            self.ops = GateList._copy_list(self.ops)
        else :
            self.ops = GateList._copy_list(ops)

    def _set_no_copy(self, ops) :
        self.ops = ops

    def append(self, op) :
        assert isinstance(op, Operator), 'GateList.append() accepts Operator.'
        self.ops.append(op.copy())
 
    def __add__(self, other) :
        if isinstance(other, GateList) :
            other = other.ops
        elif isinstance(other, Operator) :
            other = [other]
        gatelist = GateList()
        gatelist.ops = GateList._copy_list(self.ops + other)
        return gatelist
                
    def __iadd__(self, other) :
        if isinstance(other, GateList) :
            self.ops += other.ops
            return self
        if isinstance(other, Operator) :
            self.ops.append(other.copy())
            return self
        self.ops += GateList._copy_list(other)
        return self


class GateListIterator :
    """ traverse operations.
    Traversing operators through nested operartor sequences.
    """
    def __init__(self, ops) :
        self.op_iter = iter(ops)
        self.op_iter_stack = [ self.op_iter ]

    def prepend(self, op) :
        # If op is not a clause, envelop them with a new clause.
        if not isinstance(op, GateList) :
            clause = GateList()
            clause._set_no_copy(op)
            op = clause
            
        # create a new frame for clause
        self.op_iter = iter(op.ops)
        self.op_iter_stack.append(self.op_iter)
    
    def next(self) :
        op = next(self.op_iter, None)
        if op is not None :
            if not isinstance(op, GateList) :
                return op
            
            # go into new clause
            self.op_iter = iter(op.ops)
            self.op_iter_stack.append(self.op_iter)
            return self.next()
        else :
            # end of iteration at current clause
            self.op_iter_stack.pop()
            if len(self.op_iter_stack) == 0 :
                return None

            self.op_iter = self.op_iter_stack[-1];
            return self.next()

def dump(ops) :
    if isinstance(ops, GateList) :
        ops = ops.ops
    for op in ops :
        idx = '-'
        if hasattr(op, 'idx') :
            idx = str(op.idx)
        print('{}: {}'.format(idx, repr(op)))
        if isinstance(op, model.IfClause) :
            dump(op.clause)
