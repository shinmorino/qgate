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
        self.lanes = dict()
        self.qstates_list = []

    def __del__(self) :
        self.qstates_list = None

    def get_n_lanes(self) :
        return len(self.lanes)

    def get_lane(self, qreg) :
        return self.lanes[qreg.id]

    def get_lane_by_id(self, id) :
        return self.lanes[id]

    def get_state_index(self, *qregs) :
        idx = 0
        for qreg in qregs :
            idx |= 1 << self.lanes[qreg.id].external
        return idx

    @property
    def states(self) :
        return StateGetter(self, null)

    @property
    def prob(self) :
        return StateGetter(self, abs2)
    
    def get_qubit_states_list(self) :
        return self.qstates_list

    def add_qregset(self, qregset, n_lanes_per_chunk, device_ids, pkg) :
        
        # initialize qubit states
        assert len(qregset) != 0, "empty qreg set."

        # create lane map and define external_lane.
        cur_n_lanes = len(self.lanes)
        for idx, qreg in enumerate(qregset) :
            external_lane = idx + cur_n_lanes
            lane = Lane(external_lane)
            self.lanes[qreg.id] = lane
            
        qstates = pkg.create_qubit_states(self.dtype)
        n_lanes = len(qregset)

        # FIXME: better strategy to allocate chunks on multi devices
        if n_lanes_per_chunk is not None :
            # multi chunk
            n_lanes_per_chunk = min(n_lanes, n_lanes_per_chunk)
        else :
            n_lanes_per_chunk = n_lanes
                
        self.processor.initialize_qubit_states(qstates, n_lanes, n_lanes_per_chunk,
                                               device_ids);
        self.qstates_list.append(qstates)
        
        #if len(device_ids) != 0 :
        #    n_devices_consumed = checked_n_lanes_per_chunk // n_lanes
        #    rotate device_ids FIXME: rotated list not returned.
        #    device_ids = device_ids[-n_devices_consumed:] + device_ids[:-n_devices_consumed]
        
        # update lanes with layout.
        for local_lane, qreg in enumerate(qregset) :
            lane = self.lanes[qreg.id]
            lane.set_qstates_layout(qstates, local_lane)

    def reset_all_qstates(self) :
        # reset all qubit states.
        for qstates in self.get_qubit_states_list() :
            self.processor.reset_qubit_states(qstates);
    
    def calc_probability(self, qreg) :
        from qgate.model import Qreg
        if not isinstance(qreg, Qreg) :
            raise RuntimeError('qreg must be an instance of class Qreg.')
        
        lane = self.get_lane(qreg)
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
                                      self.lanes.values(), self.qstates_list,
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
                                  self.lanes.values(), self.qstates_list,
                                  1, idx, 1)

        return values[0]
