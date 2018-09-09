import qasm_model as qasm
import sim_model as sim

class Simulator :
    def __init__(self, circuits) :
        self.circuits = circuits

    def prepare(self) :
        qubit_groups = []
        ops = []
        for circuit in self.circuits :
            qubit_groups.append(sim.QubitStates(circuit.get_n_qregs()))
            ops += [(op, circuit) for op in circuit.ops] 
            
        # FIXME: sort ops
        self.ops = ops
        self.qubit_groups = qubit_groups

        self.step_iter = iter(self.ops)

    def run_step(self) :
        try :
            op = next(self.step_iter)
            self._apply_op(op[0], op[1])
            return True
        except StopIteration :
            return False

    def terminate(self) :
        pass
        
    def _apply_op(self, op, circuit) :
        if isinstance(op, qasm.Measure) :
            self._measure(op, circuit)
        elif op.get_n_inputs() == 1 :
            self._apply_unary_gate(op, circuit)
        else :
            assert op.get_n_inputs() == 2
            self._apply_binary_gate(op, circuit)
            

    def _measure(self, op, circuit) :
        pass

    def _apply_unary_gate(self, op, circuit) :
        pass

    def _apply_binary_gate(self, op, circuit) :
        pass


def py(circuits) :
    return Simulator(circuits)
