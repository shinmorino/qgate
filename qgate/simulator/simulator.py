import qgate.model as model
import qgate.model.gate as gate
from .qubits import Qubits, Lane
from .value_store import ValueStore
import numpy as np
import math
import random

def _op_key(op_tuple) :
    return op_tuple[0].idx


class Simulator :
    def __init__(self, defpkg, dtype) :
        self.defpkg = defpkg
        self.processor = defpkg.create_qubit_processor(dtype)
        self._qubits = Qubits(self.processor, dtype)

    def set_circuits(self, circuits) :
        if len(circuits) == 0 :
            raise RuntimeError('empty circuits')
        self.circuits = circuits
        
    @property    
    def qubits(self) :
        return self._qubits
    @property
    def values(self) :
        return self._value_store

    def prepare(self, n_lanes_per_chunk = None, device_ids = []) :
        self.processor.clear() # FIXME: clarify the role of clear().

        # merge all gates, and sort them.
        ops = []
        for circuit_idx, circuit in enumerate(self.circuits) :
            ops += [(op, circuit_idx) for op in circuit.ops]
        # FIXME: refine operator ordering
        ops = sorted(ops, key = _op_key)
        self.ops = ops

        # create lane map and define external_lane.
        lanes = {}
        for external_lane, qreg in enumerate(self.circuits.qregset) :
            lane = Lane(external_lane)
            lanes[qreg.id] = lane
        self._qubits.set_lanes(lanes)
        
        # initialize qubit states
        for circuit in self.circuits :
            assert len(circuit.qregset) != 0, "empty qreg set."
            
            qstates = self.defpkg.create_qubit_states(self._qubits.dtype)
            n_lanes = len(circuit.qregset)
            
            # multi chunk
            if n_lanes_per_chunk is None :
                n_lanes_per_chunk = n_lanes
            else :
                n_lanes_per_chunk = min(n_lanes, n_lanes_per_chunk)
                
            self.processor.initialize_qubit_states(qstates, n_lanes, n_lanes_per_chunk, device_ids);
            self._qubits.add_qubit_states(qstates)
            # update lanes with layout.
            for local_lane, qreg in enumerate(circuit.qregset) :
                lane = lanes[qreg.id]
                lane.set_qstates_layout(qstates, local_lane)                

        self.processor.prepare()  # prepare() could be useless.
        # reset all qubit states.
        for qstates in self._qubits.get_qubit_states_list() :
            self.processor.reset_qubit_states(qstates);
        # creating values store for references
        self._value_store = ValueStore()
        self._value_store.add(self.circuits.refset)

        # storage for previously measured value (used on reset)
        self.qreg_values = dict()
        for qreg in self.circuits.qregset :
            self.qreg_values[qreg.id] = -1
        
        self.step_iter = iter(self.ops)

    def run_step(self) :
        try :
            op_circ = next(self.step_iter)
            op = op_circ[0]
            self._apply_op(op)
            return True
        except StopIteration :
            return False

    def run(self) :
        while self.run_step() :
            pass

    def terminate(self) :
        # release resources.
        self.circuits = None
        self._value_store = None
        self.ops = None
        self._qubits = None
        
    def _apply_op(self, op) :
        if isinstance(op, model.Clause) :
            self._apply_clause(op)
        elif isinstance(op, model.IfClause) :
            self._apply_if_clause(op)
        elif isinstance(op, model.Measure) :
            self._measure(op)
        elif isinstance(op, model.Gate) :
            if op.cntrlist is None :
                if not isinstance(op.gate_type, gate.ID) : # nop
                    self._apply_unary_gate(op)
            else :
                self._apply_control_gate(op)
        elif isinstance(op, model.Barrier) :
            pass  # Since this simulator runs step-wise, able to ignore barrier.
        elif isinstance(op, model.Reset) :
            self._apply_reset(op)
        else :
            assert False, "Unknown operator."

    def _apply_if_clause(self, op) :
        if self._value_store.get_packed_value(op.refs) == op.val :
            self._apply_op(op.clause)            

    def _apply_clause(self, op) :
        for clause_op in op.ops :
            self._apply_op(clause_op)
    
    def _measure(self, op) :
        rand_num = random.random()
        lane = self._qubits.get_lane(op.qreg)
        qstates = lane.qstates
        result = self.processor.measure(rand_num, qstates, lane.local)
        self._value_store.set(op.outref, result)
        self.qreg_values[op.qreg.id] = result

    def _apply_reset(self, op) :
        for qreg in op.qregset :
            bitval = self.qreg_values[qreg.id]
            if bitval == -1 :
                raise RuntimeError('Qubit is not measured.')
            if bitval == 1 :
                lane = self._qubits.get_lane(qreg)
                qstates = lane.qstates
                self.processor.apply_reset(qstates, lane.local)

            self.qreg_values[qreg.id] = -1
                    
    def _apply_unary_gate(self, op) :
        assert len(op.qreglist) == 1, '1 qubit gate must have one qreg as the operand.' 
        lane = self._qubits.get_lane(op.qreglist[0])
        qstates = lane.qstates
        self.processor.apply_unary_gate(op.gate_type, op.adjoint, qstates, lane.local)

    def _apply_control_gate(self, op) :
        # FIXME: len(op.cntrlist) == 1 : 'multiple control qubits' is not supported.'
        assert len(op.cntrlist) == 1
        # print(op.qreglist)
        
        target_lane = self._qubits.get_lane(op.qreglist[0])
        local_control_lanes = [self._qubits.get_lane(ctrlreg).local for ctrlreg in op.cntrlist]
        qstates = target_lane.qstates # FIXME: lane.qstate will differ between lanes in future.
        self.processor.apply_control_gate(op.gate_type, op.adjoint,
                                          qstates, local_control_lanes, target_lane.local)
