import qgate.model as model
from qgate.model.pseudo_operator import FrameBegin, FrameEnd

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
        
class Barrier :
    pass

class ReferencedObservable(Observable) :
    def __init__(self, qstates, lane) :
        self.qstates = qstates
        self.lane = lane

    def set_observer(self, obs) :
        self.observer = obs

    def set(self, value) :
        self.observer.set_value(value)
        
    def get_observer(self) :
        return self.observer
    
class MeasureZ(ReferencedObservable) :
    def __init__(self, qstates, lane) :
        ReferencedObservable.__init__(self, qstates, lane)
        
class Prob(ReferencedObservable) :
    def __init__(self, qstates, lane) :
        ReferencedObservable.__init__(self, qstates, lane)

class GetState(Observable) :
    def __init__(self, qstates, lane, mathop) :
        self.qstates = qstates
        self.lane = lane
        self.mathop = mathop
        
# class Entangle
# class AddQubit
# class RemoveQubit (Reset ?? )

class Translator :
    def __init__(self, qubits) :
        self._qubits = qubits

    def __call__(self, op) :
        if isinstance(op, model.Measure) :
            return self._translate_measure(op)
        if isinstance(op, model.Prob) :
            return self._translate_prob(op)
        elif isinstance(op, model.Gate) :
            if op.ctrllist is None :
                # FIXME: ID gate should be removed during optimization.
                return self._translate_gate(op)
            else :
                return self._translate_control_gate(op)
        elif isinstance(op, model.Barrier) :
            return Barrier()
        elif isinstance(op, model.Reset) :
            return self._translate_reset(op)
        elif isinstance(op, (FrameBegin, FrameEnd)) :
            return op
        elif isinstance(op, (model.Clause, model.IfClause)) :
            assert False, 'No runtime operator for if_clause and clause.'
        elif isinstance(op, model.ComposedGate) :
            assert False, 'No runtime operator for composed gate.'
        
        assert False, "Unknown operator."

    def _translate_measure(self, op) :
        lane = self._qubits.lanes.get(op.qreg)
        return MeasureZ(lane.qstates, lane.local)
        
    def _translate_prob(self, op) :
        lane = self._qubits.lanes.get(op.qreg)
        return Prob(lane.qstates, lane.local)

    def _translate_reset(self, op) :
        # decompose model.Reset.
        # Each rop.Reset has only one lane for its argument.
        ops = []
        for qreg in  op.qregset :
            lane = self._qubits.lanes.get(qreg)
            reset = Reset(lane.qstates, lane.local)
            ops.append(reset)
        return ops
                    
    def _translate_gate(self, op) :
        assert len(op.qreglist) == 1, '1 qubit gate must have one qreg as the operand.' 
        lane = self._qubits.lanes.get(op.qreglist[0])
        qstates = lane.qstates
        return Gate(qstates, op.gate_type, op.adjoint, lane.local)

    def _translate_control_gate(self, op) :
        target_lane = self._qubits.lanes.get(op.qreglist[0])
        local_control_lanes = [self._qubits.lanes.get(ctrlreg).local for ctrlreg in op.ctrllist]
        qstates = target_lane.qstates # FIXME: lane.qstate will differ between lanes in future.
        return ControlledGate(qstates,
                              local_control_lanes, op.gate_type, op.adjoint, target_lane.local)
