from . import glue
import numpy as np


# math operations
def null(v) :
    return v
def abs2(v) :
    return v.real ** 2 + v.imag ** 2
prob = abs2


class Lane :
    def __init__(self, external) :
        self.external = external
        self.local = -1

    def set_qstates_layout(self, qstates, local) :
        self.qstates = qstates
        self.local = local


def qproc(qstates) :
    """ get qubit processor instance associated with qubit states. """
    return qstates._qproc


class StateGetter :
    def __init__(self, qubits, mathop) :
        import weakref
        self._qubits = weakref.ref(qubits)
        self._mathop = mathop
    
    def __getitem__(self, key) :
        qubits = self._qubits()
        if qubits is None :
            return None
        return qubits.get_states(self._mathop, key)


class Qubits :
    def __init__(self, dtype) :
        self.dtype = dtype
        self.qstates_list = []
        self.states = StateGetter(self, null)
        self.prob = StateGetter(self, abs2)

    def __del__(self) :
        self.qstates_list.clear()

    def get_n_qubits(self) :
        return len(self.lanes)

    def get_lane(self, qreg) :
        return self.lanes[qreg.id]
    
    def set_lanes(self, lanes) :
        self.lanes = lanes

    def get_state_index(self, *qregs) :
        idx = 0
        for qreg in qregs :
            idx |= 1 << self.lanes[qreg.id].external
        return idx
    
    def add_qubit_states(self, qstates) :
        self.qstates_list.append(qstates)
    def prepare(self) :
        all_qstates = self.get_qubit_states()
        procs = set([qproc(qstates) for qstates in all_qstates])
        assert len(procs) == 1, "Only 1 proc in qubits is allowed."
        self._proc = procs.pop()
        

    def get_qubit_states_list(self) :
        return self.qstates_list

    def calc_probability(self, qreg) :
        from qgate.model import Qreg
        if not isinstance(qreg, Qreg) :
            raise RuntimeError('qreg must be an instance of class Qreg.')
        
        prob = 1.
        lane = self.get_lane(qreg)
        proc = qproc(lane.qstates)
        return proc.calc_probability(lane.qstates, lane.local)
    
    
    def get_states(self, mathop = null, key = None) :
        if mathop == null :
            dtype = np.complex64 if self.dtype == np.float32 else np.complex128
        elif mathop == abs2 :
            dtype = self.dtype
        else :
            raise RuntimeError('unsupported mathop, {}'.format(repr(mathop)))
            
            
        n_states = 1 << self.get_n_qubits()
        if key is None :
            key = slice(0, n_states)
            
        if isinstance(key, slice) :
            start, stop, step = key.start, key.stop, key.step
            if step is None :
                step = 1
            if step == 0 :
                raise ValueError('slice step cannot be zero')
            
            if 0 < step :
                if start is None :
                    start = 0
                elif start < 0 :
                    start += n_states
                if stop is None :
                    stop = n_states
                elif stop < 0 :
                    stop += n_states
                # clip
                start = max(0, min(start, n_states))
                stop = max(0, min(stop, n_states))
                # empty range
                stop = max(start, stop)
            else :
                if start is None :
                    start = n_states - 1
                elif start < 0 :
                    start += n_states
                if stop is None :
                    stop = -1
                elif stop < 0 :
                    stop += n_states
                # clip
                start = max(-1, min(start, n_states - 1))
                stop = max(-1, min(stop, n_states - 1))
                # empty range
                stop = min(start, stop)

            n_states = (abs(stop - start + step) - 1) // abs(step)
            # print(start, stop, step, n_states)
            if n_states == 0 :
                return np.empty([0], dtype)
            values = np.empty([n_states], dtype)
            self._proc.get_states(values, 0, mathop,
                                  self.lanes.values(), self.get_qubit_states(), self.get_n_qubits(),
                                  n_states, start, step)
            return values
        
        # key is integer 
        try :
            idx = int(key)
        except :
            raise

        if idx < 0 :
            if idx <= - n_states:
                raise ValueError('list index out of range')
            idx += n_states

        if n_states <= idx :
            raise RuntimeError('list index out of range')
        
        values = np.empty([1], dtype)
        self._proc.get_states(values, 0, mathop,
                              self.lanes.values(), self.get_qubit_states(), self.get_n_qubits(),
                              1, idx, 1)

        return values[0]
