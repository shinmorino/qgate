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
        
class Barrier :
    pass

class MeasureZ(Observable) :
    def __init__(self, qstates, lane) :
        self.qstates = qstates
        self.lane = lane

    def set_observer(self, obs) :
        self.observer = obs

    def set(self, value) :
        self.observer.set_value(value)
        
    def get_observer(self) :
        return self.observer
        

class GetState(Observable) :
    def __init__(self, qtates, lane, mathop) :
        self.qstates = qstates
        self.lane = lane
        self.mathop = mathop


class FrameBegin :
    pass

class FrameEnd :
    pass
        
# class Entangle
# class AddQubit
# class RemoveQubit (Reset ?? )

class Translator :
    def __init__(self, qubits) :
        self._qubits = qubits

    def __call__(self, op) :
        if isinstance(op, model.Measure) :
            return self._translate_measure(op)
        elif isinstance(op, model.Gate) :
            if op.cntrlist is None :
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
        
        assert False, "Unknown operator."

    def _translate_measure(self, op) :
        lane = self._qubits.get_lane(op.qreg)
        return MeasureZ(lane.qstates, lane.local)
        
    def _translate_reset(self, op) :
        # FIXME: should reset have only one operand ?
        ops = []
        for qreg in  op.qregset :
            lane = self._qubits.get_lane(qreg)
            reset = Reset(lane.qstates, lane.local)
            ops.append(reset)
        return ops
                    
    def _translate_gate(self, op) :
        assert len(op.qreglist) == 1, '1 qubit gate must have one qreg as the operand.' 
        lane = self._qubits.get_lane(op.qreglist[0])
        qstates = lane.qstates
        return Gate(qstates, op.gate_type, op.adjoint, lane.local)

    def _translate_control_gate(self, op) :
        # FIXME: len(op.cntrlist) == 1 : 'multiple control qubits' is not supported.'
        assert len(op.cntrlist) == 1
        # print(op.qreglist)
        
        target_lane = self._qubits.get_lane(op.qreglist[0])
        local_control_lanes = [self._qubits.get_lane(ctrlreg).local for ctrlreg in op.cntrlist]
        qstates = target_lane.qstates # FIXME: lane.qstate will differ between lanes in future.
        return ControlledGate(qstates,
                              local_control_lanes, op.gate_type, op.adjoint, target_lane.local)
