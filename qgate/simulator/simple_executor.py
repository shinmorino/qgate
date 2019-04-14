import qgate.model as model
from .runtime_operator import Observable, Gate, ControlledGate, Reset, MeasureZ, Prob
from .runtime_operator import Translator, Observer
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
        self.translate = Translator(qubits)

    def observer(self, value_setter) :
        return SimpleObserver(self, value_setter)

    def enqueue(self, ops) :
        if isinstance(ops, list) :
            for op in ops :
                self.enqueue(op)
            return

        op = ops
        if isinstance(op, (model.NewQreg, model.ReleaseQreg, model.Cohere, model.Decohere)) :
            self.queue.append(op)
            self.flush()  # flush here to update qubits layout.
        else :
            rop = self.translate(op)
            if isinstance(rop, (MeasureZ, Prob)) :
                # set observer to value store.
                value_setter = ValueStoreSetter(self._value_store, op.outref)
                obs = self.observer(value_setter)
                rop.set_observer(obs)
                self._value_store.set(op.outref, obs)
            self.queue.append(rop)

    def flush(self) :
        while len(self.queue) != 0 :
            self.dispatch()

    def dispatch(self) :
        # get next rop
        rop = self.queue.pop(0);

        # dispatch qreg layer ops
        if isinstance(rop, model.NewQreg) :
            self._qubits.add_qubit_states([rop.qreg])
        elif isinstance(rop, model.ReleaseQreg) :
            self._qubits.deallocate_qubit_states(rop.qreg)
        elif isinstance(rop, model.Cohere) :
            self._qubits.cohere(rop.qreglist)
        elif isinstance(rop, MeasureZ) :
            if not self._qubits.lanes.exists(rop.op.qreg) :
                # target qreg does not exist.  It may happen if a qreg is used in a if clause.
                self._qubits.add_qubit_states([rop.op.qreg])

            randnum = random.random()
            lane = self._qubits.lanes.get(rop.op.qreg)
            qstates, local_lane = lane.qstates, lane.local
            prob = self.processor.calc_probability(qstates, local_lane)
            result = 0 if randnum < prob else 1
            rop.set(result)
            qstates.set_lane_state(local_lane, result)

            if len(self.queue) != 0 and isinstance(self.queue[0], model.Decohere) :
                # process decohere
                decohere = self.queue.pop(0)
                assert decohere.qreg == rop.op.qreg
                if qstates.get_n_lanes() == 1 :
                    self.processor.set_bit(result, prob, qstates, local_lane)
                else :
                    self._qubits.decohere(rop.op.qreg, result, prob)
            else :
                self.processor.set_bit(result, prob, qstates, local_lane)

        elif isinstance(rop, Prob) :
            lane = self._qubits.lanes.get(rop.op.qreg)
            result = self.processor.calc_probability(lane.qstates, lane.local)
            rop.set(result)
        elif isinstance(rop, (model.Barrier, model.ClauseBegin, model.ClauseEnd)) :
            pass

        # dispatch lane layer ops
        elif isinstance(rop, Gate) :
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
        else :
            assert False, 'Unknown operator, {}.'.format(repr(rop))
