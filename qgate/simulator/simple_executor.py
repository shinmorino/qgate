import qgate.model as model
from .runtime_operator import Observable, Gate, ControlledGate, Reset, Prob, Decohere
from .runtime_operator import Observer
from .value_store import ValueStoreSetter
import random

class ValueObserver(Observer) :
    def __init__(self, executor) :
        self._observed = False
        self.executor = executor

    @property
    def observed(self) :
        return self._observed

    def wait(self) :
        while not self.observed :
            self.executor.dispatch()

    def set_value(self, value) :
        self._observed = True
        self.value = value

    def get_value(self) :
        assert self._observed
        return self.value


class DelegatingObserver(Observer) :
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
    def __init__(self, processor, qubits, value_store) :
        self.queue = []
        self.processor = processor
        self._qubits = qubits
        self._value_store = value_store

    def value_observer(self) :
        return ValueObserver(self)

    def delegating_observer(self, value_setter) :
        return DelegatingObserver(self, value_setter)

    def enqueue(self, op) :
        self.queue.append(op)
        if isinstance(op, (model.NewQreg, model.ReleaseQreg, model.Join, model.Separate)) :
            self.flush()  # flush here to update qubits layout.

    def flush(self) :
        while len(self.queue) != 0 :
            self.dispatch()

    def dispatch(self) :
        # get next rop
        op = self.queue.pop(0);

        # dispatch qreg layer ops
        if isinstance(op, model.NewQreg) :
            self._qubits.add_qubit_states([op.qreg])
        elif isinstance(op, model.ReleaseQreg) :
            self._qubits.deallocate_qubit_states(op.qreg)
        elif isinstance(op, model.Join) :
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

            # rop execution
            prob = self.processor.calc_probability(rop_prob.qstates, rop_prob.lane)
            rop_prob.set(prob)
            # synchronized here.

            # check if Seperate and decohere can be fused.
            fuse_seperate = len(self.queue) != 0 and isinstance(self.queue[0], model.Separate)
            if fuse_seperate :
                separate = self.queue.pop(0)
                assert separate.qreg == op.qreg
                fuse_seperate = rop_prob.qstates.get_n_lanes() != 1

            randnum = random.random()
            if fuse_seperate :
                # FIXME: qreg layer, barrier here.
                result = 0 if randnum < prob else 1
                self._value_store.set(op.outref, result)
                rop_prob.qstates.set_lane_state(rop_prob.lane, result)
                self._qubits.decohere_and_separate(op.qreg, result, prob)
            else :
                # prep decohere rop.
                value_setter = ValueStoreSetter(self._value_store, op.outref)
                res_obs = self.delegating_observer(value_setter)
                self._value_store.set(op.outref, res_obs)
                decohere = Decohere(randnum, prob_obs, rop_prob.qstates, rop_prob.lane)
                decohere.set_observer(res_obs)
                # rop level execution
                result = 0 if decohere.randnum < prob else 1
                decohere.set(result)
                prob = decohere.prob_obs.get_value()
                decohere.qstates.set_lane_state(decohere.lane, result)
                self.processor.decohere(result, prob, decohere.qstates, decohere.lane)

        elif isinstance(op, model.Prob) :
            # set observer to value store.
            lane = self._qubits.lanes.get(op.qreg)
            rop = Prob(lane.qstates, lane.local)
            value_setter = ValueStoreSetter(self._value_store, op.outref)
            obs = self.delegating_observer(value_setter)
            rop.set_observer(obs)
            self._value_store.set(op.outref, obs)

            # rop execution
            lane = self._qubits.lanes.get(op.qreg)
            result = self.processor.calc_probability(lane.qstates, lane.local)
            rop.set(result)
            
        # operators that currently do not have any effects.
        elif isinstance(op, model.Barrier) :
            self.flush()
        elif isinstance(op, (model.ClauseBegin, model.ClauseEnd)) :
            pass
        # Gate ops
        elif isinstance(op, model.Gate) :
            lane = self._qubits.lanes.get(op.qreg)
            if op.ctrllist is None :
                rop = Gate(lane.qstates, op.gate_type, op.adjoint, lane.local)
                # rop execution
                self.processor.apply_unary_gate(rop.gate_type, rop.adjoint, rop.qstates, rop.lane)
            else :
                target_lane = self._qubits.lanes.get(op.qreg)
                local_control_lanes = [self._qubits.lanes.get(ctrlreg).local for ctrlreg in op.ctrllist]
                qstates = target_lane.qstates # lane.qstate must be the same for all control and target lanes.
                rop = ControlledGate(qstates,
                                     local_control_lanes, op.gate_type, op.adjoint, target_lane.local)
                
                # rop execution
                self.processor.apply_control_gate(rop.gate_type, rop.adjoint,
                                                  rop.qstates, rop.control_lanes, rop.target_lane)
        elif isinstance(op, model.Reset) :
            # FIXME: qregset
            lane = self._qubits.lanes.get(*op.qregset)
            rop = Reset(lane.qstates, lane.local)
            # rop execution
            qstates = rop.qstates
            bitval = qstates.get_lane_state(rop.lane)
            if bitval == -1 :
                raise RuntimeError('Qubit is not measured.')
            if bitval == 1 :
                self.processor.apply_reset(qstates, rop.lane)
                qstates.set_lane_state(rop.lane, -1)
        else :
            assert False, 'Unknown operator, {}.'.format(repr(rop))
