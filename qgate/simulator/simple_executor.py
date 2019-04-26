import qgate.model as model
from .runtime_operator import Observable, Gate, ControlledGate, Reset, MeasureZ, Prob
from .runtime_operator import Observer
from .value_store import ValueStoreSetter
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
    def __init__(self, processor, qubits, value_store) :
        self.queue = []
        self.processor = processor
        self._qubits = qubits
        self._value_store = value_store

    def observer(self, value_setter) :
        return SimpleObserver(self, value_setter)

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
            randnum = random.random()
            rop = MeasureZ(randnum, lane.qstates, lane.local)

            # set observer to value store.
            value_setter = ValueStoreSetter(self._value_store, op.outref)
            obs = self.observer(value_setter)
            rop.set_observer(obs)
            self._value_store.set(op.outref, obs)

            # rop execution
            qstates, local_lane = rop.qstates, rop.lane
            prob = self.processor.calc_probability(qstates, local_lane)
            # synchronized here.
            result = 0 if rop.randnum < prob else 1
            rop.set(result)

            qstates.set_lane_state(local_lane, result)

            # fuse Seperate and decohere if possible
            merge_seperate = len(self.queue) != 0 and isinstance(self.queue[0], model.Separate)
            if fuse_seperate :
                separate = self.queue.pop(0)
                assert separate.qreg == op.qreg
                fuse_seperate = qstates.get_n_lanes() == 1

            if fuse_seperate :
                # FIXME: qreg layer, barrier here.
                # process separate
                self._qubits.decohere_and_separate(op.qreg, result, prob)
            else :
                # rop level execution
                self.processor.decohere(result, prob, qstates, local_lane)

        elif isinstance(op, model.Prob) :
            # set observer to value store.
            lane = self._qubits.lanes.get(op.qreg)
            rop = Prob(lane.qstates, lane.local)
            value_setter = ValueStoreSetter(self._value_store, op.outref)
            obs = self.observer(value_setter)
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
