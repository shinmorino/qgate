from .runtime_operator import Observable, Gate, ControlledGate, Reset, Barrier, MeasureZ, Observer
from qgate.model.pseudo_operator import FrameBegin, FrameEnd
import random

class SimpleObserver(Observer) :
    def __init__(self, executor, value_setter) :
        self._observed = False
        self.executor = executor
        self.value_setter = value_setter

    @property
    def observed(self) :
        return self._observed

    def wait(self) :
        while not self.observed :
            self.executor.dispatch()

    def set_value(self, value) :
        self._observed = True
        self.value_setter(value)


class SimpleExecutor :
    def __init__(self, processor) :
        self.queue = []
        self.processor = processor

    def observer(self, value_setter) :
        return SimpleObserver(self, value_setter)
        
    def enqueue(self, rops) :
        if isinstance(rops, list) :
            for rop in rops :
                self.enqueue(rop)
            return
        # rops is not a list 
        self.queue.append(rops)
        if isinstance(rops, Observable) :
            # FIXME: optimize rops in queue here
            observer = rops.get_observer()
            while not observer.observed :
                self.dispatch()
            
    def flush(self) :
        while self.dispatch() :
            pass

    def dispatch(self) :
        # all rops have been processed.
        if len(self.queue) == 0 :
            return False

        # get next rop
        rop = self.queue[0];
        self.queue.remove(rop)

        # dispatch
        if isinstance(rop, Gate) :
            self.processor.apply_unary_gate(rop.gate_type, rop.adjoint, rop.qstates, rop.lane)
        elif isinstance(rop, ControlledGate) :
            self.processor.apply_control_gate(rop.gate_type, rop.adjoint,
                                              rop.qstates, rop.control_lanes, rop.target_lane)
        elif isinstance(rop, Reset) :
            qstates = rop.qstates
            bitval = qstates.get_lane_state(rop.lane)
            if bitval == -1 :
                raise RuntimeError('Qubit is not measured.')
            if bitval == 1 :
                self.processor.apply_reset(qstates, rop.lane)
                qstates.set_lane_state(rop.lane, -1)
        elif isinstance(rop, MeasureZ) :
            rand_num = random.random()
            result = self.processor.measure(rand_num, rop.qstates, rop.lane)
            rop.set(result)
            rop.qstates.set_lane_state(rop.lane, result)
        elif isinstance(rop, (Barrier, FrameBegin, FrameEnd)) :
            pass
        else :
            assert False, 'Unknown operator, {}.'.format(repr(rop))

        return True
