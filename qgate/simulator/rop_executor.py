from .runtime_operator import ControlledGate, Gate, Reset, Prob, Decohere, ReferencedObservable

class RopExecutor :
    def __init__(self, processor) :
        self.processor = processor
        self.queue = list()

    def enqueue(self, rop) :
        self.queue.append(rop)

    def flush(self) :
        while len(self.queue) != 0 :
            self.dispatch()

    def wait_rop(self, rop) :
        try :
            idx = self.queue.index(rop)
            for _ in range(idx + 1) :
                self.dispatch()
        except :
            pass

    def wait_observable(self, observer) :
        observables = [ op for op in self.queue if isinstance(op, ReferencedObservable) ]
        for observable in observables :
            if observer == observable.get_observer() :
                self.wait_rop(observable)

    def dispatch(self) :
        rop = self.queue.pop(0)
        if isinstance(rop, ControlledGate) :
            self.processor.apply_control_gate(rop.gate_type, rop.adjoint,
                                              rop.qstates, rop.control_lanes, rop.target_lane)
        elif isinstance(rop, Gate) :
            self.processor.apply_unary_gate(rop.gate_type, rop.adjoint, rop.qstates, rop.lane)
        elif isinstance(rop, Prob) :
            prob = self.processor.calc_probability(rop.qstates, rop.lane)
            rop.set(prob)
        elif isinstance(rop, Decohere) :
            prob = rop.prob_obs.get_value()
            result = 0 if rop.randnum < prob else 1
            rop.set(result)
            prob = rop.prob_obs.get_value()
            rop.qstates.set_lane_state(rop.lane, result)
            self.processor.decohere(result, prob, rop.qstates, rop.lane)
        elif isinstance(rop, Reset) :
            bitval = rop.qstates.get_lane_state(rop.lane)
            if bitval == -1 :
                raise RuntimeError('Qubit is not measured.')
            if bitval == 1 :
                self.processor.apply_reset(rop.qstates, rop.lane)
                rop.qstates.set_lane_state(rop.lane, -1)
        else :
            assert False, 'Unknown rop, {}.'.format(repr(rop))
