from . import model
from . import pseudo_operator
from . import gatelist
from .qreg_aggregator import QregAggregator
from .decompose import decompose

class Preprocessor :
    def __init__(self) :
        self.aggregator = QregAggregator()
        self.reset()
    
    def reset(self) :
        self.aggregator.reset()
        self._refset = set()
        
    def get_qregset(self) :
        return self.aggregator.qregset
    
    def get_qregsetlist(self) :
        return self.aggregator.qregsetlist

    def get_refset(self) :
        return self._refset

    def preprocess_operator_list(self, oplist) :
        preprocessed = list()
        for op in oplist :
            if isinstance(op, model.Gate) :
                # 1 qubit gate
                self.aggregator.add_qreg(op.qreg)
                if not op.ctrllist is None :
                    for qreg in op.ctrllist :
                        self.aggregator.add_qreg(qreg)
                    self.aggregator.aggregate([op.qreg] + op.ctrllist)                    
                preprocessed.append(op)
            elif isinstance(op, (model.Measure, model.Prob)) :
                # Measure, Prob
                self.aggregator.add_qreg(op.qreg)
                self._refset.add(op.outref)
                preprocessed.append(op)
            elif isinstance(op, (model.Reset, model.Barrier)) :
                # Reset, Barrier
                factory = op.__class__
                for qreg in op.qregset :
                    overlap = self.aggregator.qregset & op.qregset
                    if len(overlap) != 0 :
                        preprocessed.append(factory({qreg}))
                    # FIXME: add warning when unused qreg found.
            elif isinstance(op, (model.MultiQubitGate, model.ComposedGate,
                                 model.PauliMeasure, model.PauliProb)) :
                # PauliMeasure, PauliProb, MultiQubitGate, ComposedGate
                expanded = decompose(op)
                preprocessed += self.preprocess_operator_list(expanded)
            elif isinstance(op, gatelist.GateList) :
                # GateList
                expanded = self.preprocess_clause(op)
                preprocessed += self.preprocess_operator_list(expanded)
            elif isinstance(op, model.IfClause) :
                # IfClause
                # FIXME: add warning when unused ref found.
                child_clause = self.preprocess_clause(op.clause)
                if_clause = model.IfClause(op.refs, op.cond, child_clause)
                preprocessed.append(if_clause)
            elif isinstance(op, (pseudo_operator.ClauseBegin, pseudo_operator.ClauseEnd)) :
                preprocessed.append(op)
            else :
                assert False, 'Unexpected op, {}'.format(repr(op))

        return preprocessed

    def preprocess_clause(self, clause) :
        preprocessed = [ pseudo_operator.ClauseBegin() ]
        preprocessed += self.preprocess_operator_list(clause.ops)
        preprocessed.append(pseudo_operator.ClauseEnd())
        return preprocessed

    def preprocess(self, clause) :
        ops = self.preprocess_clause(clause)
        
        # give operators their numbers.
        for idx, op in enumerate(ops) :
            op.set_idx(idx)

        preprocessed = gatelist.GateList()
        preprocessed.ops = ops
        return preprocessed
