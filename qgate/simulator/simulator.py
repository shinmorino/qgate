from .qubits import Qubits
from .value_store import ValueStore, ValueStoreSetter
import qgate.model as model
from qgate.model.gatelist import GateListIterator
from .simple_executor import SimpleExecutor
from .runtime_operator import Translator, Observer
import numpy as np
import math
import copy


class Simulator :
    def __init__(self, defpkg, **prefs) :
        dtype = prefs.get('dtype', np.float64)
        self.defpkg = defpkg
        self.preprocessor = model.Preprocessor()
        self.processor = defpkg.create_qubit_processor(dtype)
        self._qubits = Qubits(self.processor, dtype)
        self.translate = Translator(self._qubits)
        self.prefs = dict()
        self.set_preference(**prefs)
        self.reset()

    def reset(self) :
        self.preprocessor.reset() # reset circuit states
        self.processor.reset() # release all internal objects
        self.executor = SimpleExecutor(self.processor)
        self._qubits.reset()

    def run(self, circuit) :
        if not isinstance(circuit, model.GateList) :
            ops = circuit
            circuit = model.GateList()
            circuit.set(ops)
            
        preprocessed = self.preprocessor.preprocess(circuit)
        self.prepare()

        self.op_iter = GateListIterator(preprocessed.ops)
        while True :
            op = self.op_iter.next()
            if op is None :
                break
            if isinstance(op, model.IfClause) :
                if self._evaluate_if(op) :
                    self.op_iter.prepend(op.clause)
            else :
                rop = self.translate(op)
                if isinstance(op, (model.Measure, model.Prob)) :
                    value_setter = ValueStoreSetter(self._value_store, op.outref)
                    # observer
                    obs = self.executor.observer(value_setter)
                    rop.set_observer(obs)
                self.executor.enqueue(rop)
                
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
        
    def set_preference(self, **prefs) :
        for k, v in prefs.items() :
            self.prefs[k] = copy.copy(v)
        
    @property    
    def qubits(self) :
        return self._qubits
    
    @property
    def values(self) :
        return self._value_store

    def prepare(self) :
        isolate_circuits = self.prefs.get('isolate_circuits', True)
        if isolate_circuits :
            for qregset in self.preprocessor.get_qregsetlist() :
                self._qubits.allocate_qubit_states(self.defpkg, qregset)
        else :
            self._qubits.allocate_qubit_states(self.defpkg, self.preprocessor.get_qregset())
                
        self._qubits.reset_all_qstates()
        
        # creating values store for references
        self._value_store = ValueStore()
        self._value_store.add(self.preprocessor.get_refset())

    def terminate(self) :
        # release resources.
        self.circuits = None
        self._value_store = None
        self.ops = None
        self._qubits = None
