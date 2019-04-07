from .qubits import Qubits
from .value_store import ValueStore
import qgate.model as model
from qgate.model.gatelist import GateListIterator
from .simple_executor import SimpleExecutor
from .runtime_operator import Observer
import numpy as np
import math
import copy


class Simulator :
    def __init__(self, defpkg, **prefs) :
        dtype = prefs.get('dtype', np.float64)
        self.defpkg = defpkg
        self.processor = defpkg.create_qubit_processor(dtype)
        self._qubits = Qubits(self.processor, dtype)
        self.prefs = dict()
        self.set_preference(**prefs)
        self._value_store = ValueStore()
        self.reset()

    @property
    def qubits(self) :
        return self._qubits

    @property
    def values(self) :
        return self._value_store

    def set_preference(self, **prefs) :
        for k, v in prefs.items() :
            self.prefs[k] = copy.copy(v)

    def reset(self) :
        # release all internal objects
        self.processor.reset()
        self._qubits.reset()
        self.executor = SimpleExecutor(self.processor, self._qubits, self._value_store)
        self.preprocessor = model.Preprocessor(**self.prefs)
        self._value_store.reset()

    def terminate(self) :
        # release resources.
        self.circuits = None
        self._value_store = None
        self.ops = None
        self._qubits = None

    def run(self, circuit) :
        if not isinstance(circuit, model.GateList) :
            ops = circuit
            circuit = model.GateList()
            circuit.set(ops)
            
        preprocessed = self.preprocessor.preprocess(circuit)
        
        # model.dump(preprocessed)

        self._value_store.sync_refs(self.preprocessor.get_refset())

        self.op_iter = GateListIterator(preprocessed.ops)
        while True :
            op = self.op_iter.next()
            if op is None :
                break
            if isinstance(op, model.IfClause) :
                if self._evaluate_if(op) :
                    self.op_iter.prepend(op.clause)
            else :
                self.executor.enqueue(op)

        self.executor.flush()


    def _evaluate_if(self, op) :
        # synchronize
        values = self._value_store.get(op.refs)
        for value in values :
            if isinstance(value, Observer) :
                value.wait()

        if callable(op.cond) :
            values = self._value_store.get(op.refs)
            return op.cond(*values)
        else :
            packed_value = self._value_store.get_packed_value(op.refs)
            return packed_value == op.cond
