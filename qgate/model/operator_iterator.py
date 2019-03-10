import qgate.model as model
from .pseudo_operator import FrameBegin, FrameEnd

class OperatorIterator :
    """ traverse operations.
    Traversing operators through nested operartor sequences.
    Before and after a sequence in a clause, FrameBegin, FrameEnd is inserted.
    """
    def __init__(self, ops) :
        self.op_iter = iter(ops)
        self.op_iter_stack = [ self.op_iter ]

    def prepend(self, op) :
        # If op is not a clause, envelop them with a new clause.
        if not isinstance(op, model.Clause) :
            clause = model.Clause()
            clause.add(op)
            op = clause
            
        # create a new frame for clause
        self.op_iter = iter([FrameBegin()] + op.ops)
        self.op_iter_stack.append(self.op_iter)
    
    def next(self) :
        op = next(self.op_iter, None)
        if op is not None :
            if not isinstance(op, model.Clause) :
                return op
            
            # go into new frame
            self.op_iter = iter(op.ops)
            self.op_iter_stack.append(self.op_iter)
            return FrameBegin()
        else :
            # end of iteratoration
            self.op_iter_stack.pop()
            if len(self.op_iter_stack) == 0 :
                return None

            self.op_iter = self.op_iter_stack[-1];
            return FrameEnd()
