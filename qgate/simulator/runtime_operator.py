import qgate.model as model

class Observable :
    pass

# deined as a base class
class Observer :
    pass

    #def wait(self) :
    #    pass

    #@property
    #def observed(self) :
    #    return None
    
    #def get_value(self) :
    #    pass
    

class ControlledGate :
    def __init__(self, qstates, control_lanes, gate_type, adjoint, target_lane) :
        self.qstates = qstates
        self.control_lanes = control_lanes
        self.gate_type = gate_type
        self.adjoint = adjoint
        self.target_lane = target_lane

class Gate :
    def __init__(self, qstates, gate_type, adjoint, lane) :
        self.qstates = qstates
        self.gate_type = gate_type
        self.adjoint = adjoint
        self.lane = lane
        
class Reset :
    def __init__(self, qstates, lane) :
        self.qstates = qstates
        self.lane = lane
        
class ReferencedObservable(Observable) :

    def set_observer(self, obs) :
        self.observer = obs

    def set(self, value) :
        self.observer.set_value(value)
        
    def get_observer(self) :
        return self.observer
    
class MeasureZ(ReferencedObservable) :
    def __init__(self, measure) :
        self.op = measure
        
class Prob(ReferencedObservable) :
    def __init__(self, prob) :
        self.op = prob

class Translator :
    def __init__(self, qubits) :
        self._qubits = qubits

    def __call__(self, op) :
        if isinstance(op, model.Measure) :
            return MeasureZ(op)
        elif isinstance(op, model.Prob) :
            return Prob(op)
        elif isinstance(op, (model.Barrier, model.ClauseBegin, model.ClauseEnd)) :
            return op
        elif isinstance(op, model.Gate) :
            if op.ctrllist is None :
                # FIXME: ID gate should be removed during optimization.
                return self._translate_gate(op)
            else :
                return self._translate_control_gate(op)
        elif isinstance(op, model.Reset) :
            return self._translate_reset(op)
        elif isinstance(op, (model.IfClause, model.ComposedGate, model.GateList)) :
            assert False, 'No runtime operator for {}.'.format(repr(op))
        
        assert False, "Unknown operator, {}.".format(repr(op))

    def _translate_reset(self, op) :
        assert len(op.qregset) == 1
        lane = self._qubits.lanes.get(*op.qregset)
        return Reset(lane.qstates, lane.local)
                    
    def _translate_gate(self, op) :
        lane = self._qubits.lanes.get(op.qreg)
        return Gate(lane.qstates, op.gate_type, op.adjoint, lane.local)

    def _translate_control_gate(self, op) :
        target_lane = self._qubits.lanes.get(op.qreg)
        local_control_lanes = [self._qubits.lanes.get(ctrlreg).local for ctrlreg in op.ctrllist]
        qstates = target_lane.qstates # lane.qstate must be the same for all control and target lanes.
        return ControlledGate(qstates,
                              local_control_lanes, op.gate_type, op.adjoint, target_lane.local)
