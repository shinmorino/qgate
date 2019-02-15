import qgate.model as model
import qgate.model.gate as gate
from .qubits import Qubits, Lane
from .value_store import ValueStore
import numpy as np
import math
import random


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
        self.processor.reset() # release all internal objects

        # merge all gates, and sort them.
        ops = []
        for circuit in self.circuits :
            ops += circuit.ops
        # FIXME: refine operator ordering
        self.ops = sorted(ops, key = lambda op : op.idx)

        for circuit in self.circuits :
            self._qubits.add_qregset(circuit.qregset, n_lanes_per_chunk, device_ids, self.defpkg)
        self._qubits.reset_all_qstates()
        
        # creating values store for references
        self._value_store = ValueStore()
        self._value_store.add(self.circuits.refset)
        
        self.step_iter = iter(self.ops)

    def run_step(self) :
        try :
            op = next(self.step_iter)
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
        self._qubits.qreg_values[op.qreg.id] = result

    def _apply_reset(self, op) :
        for qreg in op.qregset :
            bitval = self._qubits.qreg_values[qreg.id]
            if bitval == -1 :
                raise RuntimeError('Qubit is not measured.')
            if bitval == 1 :
                lane = self._qubits.get_lane(qreg)
                qstates = lane.qstates
                self.processor.apply_reset(qstates, lane.local)

            self._qubits.qreg_values[qreg.id] = -1
                    
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
