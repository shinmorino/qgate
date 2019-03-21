import numpy as np
from . import lanes

# math operations
def null(v) :
    return v
def abs2(v) :
    return v.real ** 2 + v.imag ** 2
prob = abs2

        

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
    def __init__(self, processor, dtype) :
        self.processor = processor
        self.dtype = dtype
        self.lanes = lanes.Lanes()
        self.qstates_list = []

    def __del__(self) :
        self.qstates_list = None

    def reset(self) :
        self.lanes.reset()
        for qstates in self.qstates_list :
            del qstates
        self.qstates_list = []

    def get_n_lanes(self) :
        return self.lanes.get_n_lanes()

    @property
    def states(self) :
        return StateGetter(self, null)

    @property
    def prob(self) :
        return StateGetter(self, abs2)
    
    def get_qubit_states_list(self) :
        return self.qstates_list

    def set_factory(self, factory) :
        self.factory = factory

    def allocate_qubit_states(self, qregset) :
        
        # initialize qubit states
        assert len(qregset) != 0, "empty qreg set."

        n_lanes = len(qregset)
        
        # allocate qubit states
        qstates = self.factory.create(n_lanes, self.dtype, self.processor)
        self.qstates_list.append(qstates)

        # sort qregset by qreg.id before lane asssignment.
        # FIXME: need better ordering definitions.
        cur_n_lanes = self.lanes.get_n_lanes()
        sorted_qreglist = sorted(qregset, key = lambda qreg:qreg.id)
        
        # create lane map and define external_lane.
        for local_lane, qreg in enumerate(sorted_qreglist) :
            external_lane = local_lane + cur_n_lanes
            self.lanes.add_lane(qreg, external_lane, qstates, local_lane)

    def reset_all_qstates(self) :
        # reset all qubit states.
        for qstates in self.get_qubit_states_list() :
            self.processor.reset_qubit_states(qstates);
    
    def calc_probability(self, qreg) :
        from qgate.model import Qreg
        if not isinstance(qreg, Qreg) :
            raise RuntimeError('qreg must be an instance of class Qreg.')
        
        lane = self.lanes.get(qreg)
        return self.processor.calc_probability(lane.qstates, lane.local)
    
    
    def get_states(self, mathop = null, key = None) :
        if mathop == null :
            dtype = np.complex64 if self.dtype == np.float32 else np.complex128
        elif mathop == abs2 :
            dtype = self.dtype
        else :
            raise RuntimeError('unsupported mathop, {}'.format(repr(mathop)))
            
        n_states = 1 << self.get_n_lanes()
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
            self.processor.get_states(values, 0, mathop,
                                      self.lanes.all(), self.qstates_list,
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
        self.processor.get_states(values, 0, mathop,
                                  self.lanes.all(), self.qstates_list,
                                  1, idx, 1)

        return values[0]
