from . import model
from .qreg_aggregator import QregAggregator

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

    def preprocess(self, clause) :
        for op in clause.ops :
            if isinstance(op, (model.Measure, model.Prob)) :
                self.aggregator.add_qreg(op.qreg)
                self._refset.add(op.outref)
            elif isinstance(op, (model.PauliMeasure, model.PauliProb)) :
                for gate in op.gatelist :
                    self.aggregator.add_qreg(gate.qreg)
            elif isinstance(op, model.Gate) :
                self.aggregator.add_qreg(op.qreg)
                if not op.ctrllist is None :
                    for qreg in op.ctrllist :
                        self.aggregator.add_qreg(qreg)
                    self.aggregator.aggregate([op.qreg] + op.ctrllist)
            elif isinstance(op, (model.Barrier, model.Reset)) :
                for qreg in op.qregset :
                    self.aggregator.add_qreg(qreg)
                    
            elif isinstance(op, model.IfClause) :
                self.preprocess(op.clause)
                
            elif isinstance(op, (FrameBegin, FrameEnd)) :
                pass
            elif isinstance(op, model.GateList) :
                self.collect_qregs(op)
            else :
                raise RuntimeError(repr(op))
        
        # give operators their numbers.
        for idx, op in enumerate(clause.ops) :
            op.set_idx(idx)
