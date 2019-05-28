import qgate.model as model
from .runtime_operator import Observable, Gate, ControlledGate, Reset, Prob, Decohere
from .runtime_operator import Observer
from .rop_executor import RopExecutor
from .value_store import ValueStoreSetter
import random

class ValueObserver(Observer) :
    def __init__(self) :
        self._observed = False

    @property
    def observed(self) :
        return self._observed

    def set_value(self, value) :
        self._observed = True
        self.value = value

    def get_value(self) :
        assert self._observed
        return self.value


class DelegatingObserver(Observer) :
    def __init__(self, value_setter) :
        self._observed = False
        self.value_setter = value_setter

    @property
    def observed(self) :
        return self._observed

    def set_value(self, value) :
        self._observed = True
        self.value_setter(value)


class ModelExecutor :
    def __init__(self, processor, qubits, value_store) :
        self.queue = []
        self.processor = processor
        self._qubits = qubits
        self._value_store = value_store
        self._rop_executor = RopExecutor(processor)

    def value_observer(self) :
        return ValueObserver()

    def delegating_observer(self, value_setter) :
        return DelegatingObserver(value_setter)

    def enqueue(self, op) :
        if isinstance(op, model.Measure) :
            self._value_store.set(op.outref, op)
        self.queue.append(op)

    def flush(self) :
        while len(self.queue) != 0 :
            self.dispatch()
        self.rop_barrier()

    def rop_barrier(self) :
        self._rop_executor.flush()

    def wait_op(self, op) :
        try :
            idx = self.queue.index(op)
            for _ in range(idx + 1) :
                self.dispatch()
        except :
            pass

    def wait_observable(self, obs) :
        observables = [ op for op in self.queue if isinstance(op, (model.Measure, model.Prob)) ]
        for observable in observables :
            if obs == observable.get_observer() :
                self.wait_op(observable)
        self._rop_executor.wait_observable(obs)

    def dispatch(self) :
        # get next rop
        op = self.queue.pop(0);

        # dispatch qreg layer ops
        if isinstance(op, model.NewQreg) :
            self._qubits.add_qubit_states([op.qreg])
        elif isinstance(op, model.ReleaseQreg) :
            self.rop_barrier()  # all operators having op.qreg should be completed.
            self._qubits.deallocate_qubit_states(op.qreg)
        elif isinstance(op, model.Join) :
            self.rop_barrier()  # all operators having qreglist should be completed.
            self._qubits.join(op.qreglist)

        # observable ops
        elif isinstance(op, model.Measure) :
            if not self._qubits.lanes.exists(op.qreg) :
                # target qreg does not exist.  It may happen if a qreg is used in a if clause.
                self._qubits.add_qubit_states([op.qreg])

            # translate
            lane = self._qubits.lanes.get(op.qreg)
            rop_prob = Prob(lane.qstates, lane.local)

            # set observer for prob.
            prob_obs = self.value_observer()
            rop_prob.set_observer(prob_obs)
            # enqueue prob rop
            self._rop_executor.enqueue(rop_prob)

            # check if Seperate and decohere can be fused.
            fuse_separate = len(self.queue) != 0 and isinstance(self.queue[0], model.Separate)
            if fuse_separate :
                separate = self.queue.pop(0)
                assert separate.qreg == op.qreg
                fuse_separate = rop_prob.qstates.get_n_lanes() != 1

            randnum = random.random()
            if fuse_separate :
                # processed synchronously, not using observer.
                self._rop_executor.wait_rop(rop_prob) # wait for prob observed.
                prob = prob_obs.get_value()
                result = 0 if randnum < prob else 1
                self._value_store.set(op.outref, result)
                rop_prob.qstates.set_lane_state(rop_prob.lane, result)
                self.rop_barrier()  # operators using op.qreg should be completed.
                # processed synchronously
                self._qubits.decohere_and_separate(op.qreg, result, prob)
            else :
                # prep decohere rop.
                value_setter = ValueStoreSetter(self._value_store, op.outref)
                res_obs = self.delegating_observer(value_setter)
                self._value_store.set(op.outref, res_obs)
                rop_decohere = Decohere(randnum, prob_obs, rop_prob.qstates, rop_prob.lane)
                rop_decohere.set_observer(res_obs)
                # enqueue decohere rop.
                self._rop_executor.enqueue(rop_decohere)

        elif isinstance(op, model.Prob) :
            # set observer to value store.
            lane = self._qubits.lanes.get(op.qreg)
            rop = Prob(lane.qstates, lane.local)
            value_setter = ValueStoreSetter(self._value_store, op.outref)
            obs = self.delegating_observer(value_setter)
            rop.set_observer(obs)
            self._value_store.set(op.outref, obs)

            # enqueue prob rop
            self._rop_executor.enqueue(rop)

        # operators that currently do not have any effects.
        elif isinstance(op, model.Barrier) :
            self.rop_barrier()
        elif isinstance(op, (model.ClauseBegin, model.ClauseEnd)) :
            pass
        # Gate ops
        elif isinstance(op, model.Gate) :
            target_lane = self._qubits.lanes.get(op.qreg)
            if op.ctrllist is None :
                rop = Gate(target_lane.qstates, op.gate_type, op.adjoint, target_lane.local)
            else :
                local_control_lanes = [self._qubits.lanes.get(ctrlreg).local for ctrlreg in op.ctrllist]
                qstates = target_lane.qstates # lane.qstate must be the same for all control and target lanes.
                rop = ControlledGate(qstates,
                                     local_control_lanes, op.gate_type, op.adjoint, target_lane.local)
            # enqueue gate / controlled gate rop.
            self._rop_executor.enqueue(rop)

        elif isinstance(op, model.Reset) :
            # FIXME: qregset
            lane = self._qubits.lanes.get(op.qreg)
            rop = Reset(lane.qstates, lane.local)
            # enqueue reset rop
            self._rop_executor.enqueue(rop)
        else :
            assert False, 'Unknown operator, {}.'.format(repr(rop))
