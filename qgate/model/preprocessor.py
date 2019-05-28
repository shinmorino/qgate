from . import model
from . import directive
from . import gatelist
from .qreg_aggregator import QregAggregator
from .decompose import decompose
from . import prefs

class Preprocessor :
    def __init__(self, **prefdict) :
        self.circ_prep = prefdict.get(prefs.circuit_prep, prefs.dynamic)
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
                joined = self.aggregator.aggregate(all_qregs)
                if joined is not None and self.dynamic :
                    # Join qregs(qubits) if they're separated.
                    preprocessed.append(directive.Join(joined))
            preprocessed.append(op)
        elif isinstance(op, (model.Measure, model.Prob)) :
            # Measure, Prob
            if self.aggregator.add_qreg(op.qreg) and self.dynamic :
                # FIXME: not required.
                preprocessed.append(directive.NewQreg(op.qreg))
            self._refset.add(op.outref)
            preprocessed.append(op)
            if isinstance(op, model.Measure) and self.dynamic:
                # insert Separate after measurement.
                self.aggregator.separate_qreg(op.qreg)
                preprocessed.append(directive.Separate(op.qreg))
        elif isinstance(op, directive.ReleaseQreg) :
            if self.dynamic :
                if not op.qreg in self.aggregator.qregset :
                    raise RuntimeError('qreg{} is not in circuit.'.format(op.qreg.id))
                qregset = self.aggregator.find_qregset(op.qreg)
                if len(qregset) != 1 :
                    raise RuntimeError('qreg{} is not separated.'.format(op.qreg.id))
                preprocessed.append(op)
            else :
                pass  # FIXME: raise error or output warning
        elif isinstance(op, model.Reset) :
            # Reset, check if qreg is already used.
            if not op.qreg in self.aggregator.qregset :
                raise RuntimeError('unused qreg found, {}'.format(qreg))
            preprocessed.append(op)
        elif isinstance(op, model.Barrier) :
            # barrier
            preprocessed.append(op)
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
                        prologue.append(directive.Join(qregset))
            else :
                # static_dumb
                qregset = self.aggregator.qregset
                if len(qregset) == 1 :
                    prologue.append(directive.NewQreg(*qregset))
                else :
                    prologue.append(directive.Join(qregset))

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
