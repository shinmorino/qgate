from . import model
from . import directive
from . import gatelist
from .qreg_aggregator import QregAggregator
from .decompose import decompose
from . import prefs

class Preprocessor :
    def __init__(self, **prefdict) :
        self.circ_prep = prefdict.get(prefs.circuit_prep, prefs.static)
        self.dynamic = self.circ_prep == prefs.dynamic
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

    def preprocess_operator(self, op) :
        preprocessed = list()
        if isinstance(op, model.Gate) :
            # Single qubit gate
            if op.ctrllist is None :
                # Single qubit gate without control bits.
                if self.aggregator.add_qreg(op.qreg) and self.dynamic :
                    # FIXME: not required.
                    preprocessed.append(directive.NewQreg(op.qreg))
            else :
                # Single qubit gate with control bits.
                all_qregs = [op.qreg] + op.ctrllist
                merged = self.aggregator.aggregate(all_qregs)
                if merged is not None and self.dynamic :
                    # Cohere qregs(qubits) if they're seperated.
                    preprocessed.append(directive.Cohere(merged))
            preprocessed.append(op)
        elif isinstance(op, (model.Measure, model.Prob)) :
            # Measure, Prob
            if self.aggregator.add_qreg(op.qreg) and self.dynamic :
                # FIXME: not required.
                preprocessed.append(directive.NewQreg(op.qreg))
            self._refset.add(op.outref)
            preprocessed.append(op)
            if isinstance(op, model.Measure) and self.dynamic:
                # insert Decohere for measured qreg.
                self.aggregator.separate_qreg(op.qreg)
                preprocessed.append(directive.Decohere(op.qreg))
        elif isinstance(op, (model.Reset, model.Barrier)) :
            # Reset, Barrier
            # check if unkown qregs used
            new_qregs = self.aggregator.qregset - op.qregset
            if len(new_qregs) != 0 :
                qregstr = [repr(qreg) for qreg in new_qregs]
                msg = ' '.join(qregstr)
                raise RuntimeError('unused qregs found, {}'.format(msg))
            # FIXME: decomposing operator to have one qreg.  May not be required.
            factory = op.__class__
            for qreg in op.qregset :
                overlap = self.aggregator.qregset & op.qregset
                if len(overlap) != 0 :
                    preprocessed.append(factory({qreg}))
        elif isinstance(op, (model.MultiQubitGate, model.ComposedGate,
                             model.PauliMeasure, model.PauliProb)) :
            # PauliMeasure, PauliProb, MultiQubitGate, ComposedGate
            expanded = decompose(op)
            preprocessed += self.preprocess_operator_list(expanded)
        elif isinstance(op, gatelist.GateList) :
            # GateList
            preprocessed += self.preprocess_clause(op)
        elif isinstance(op, model.IfClause) :
            # IfClause
            # FIXME: add warning when unused ref found.
            child_clause = self.preprocess_clause(op.clause)
            if_clause = model.IfClause(op.refs, op.cond, child_clause)
            preprocessed.append(if_clause)
        elif isinstance(op, (directive.ClauseBegin, directive.ClauseEnd)) :
            preprocessed.append(op)
        else :
            assert False, 'Unexpected op, {}'.format(repr(op))

        return preprocessed

    def preprocess_operator_list(self, oplist) :
        preprocessed = list()
        for op in oplist :
            preprocessed += self.preprocess_operator(op)
        return preprocessed

    def preprocess_clause(self, clause) :
        preprocessed = [ directive.ClauseBegin() ]
        preprocessed += self.preprocess_operator_list(clause.ops)
        preprocessed.append(directive.ClauseEnd())
        return preprocessed

    def preprocess(self, clause) :
        ops = self.preprocess_clause(clause)

        if not self.dynamic :
            prologue = list()
            if self.circ_prep == 'static' :
                for qregset in self.aggregator.qregsetlist :
                    if len(qregset) == 1 :
                        prologue.append(directive.NewQreg(*qregset))
                    else :
                        prologue.append(directive.Cohere(qregset))
            else :
                # static_dumb
                qregset = self.aggregator.qregset
                if len(qregset) == 1 :
                    prologue.append(directive.NewQreg(*qregset))
                else :
                    prologue.append(directive.Cohere(qregset))

            ops = prologue + ops

        # set execution order of operators
        it = gatelist.GateListIterator(ops)
        idx = 0
        while True :
            op = it.next()
            if op is None :
                break
            op.set_idx(idx)
            idx += 1
            if isinstance(op, model.IfClause) :
                it.prepend(op.clause)

        preprocessed = gatelist.GateList()
        preprocessed.ops = ops
        return preprocessed
