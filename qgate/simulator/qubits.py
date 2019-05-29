import numpy as np
from . import lanes

# math operations
def null(v) :
    return v
def abs2(v) :
    return v.real ** 2 + v.imag ** 2
prob = abs2

        
# FIXME: implement __iter__ method.
class StateGetter :
    def __init__(self, qubits, mathop, n_lanes) :
        import weakref
        self._qubits = weakref.ref(qubits)
        self._mathop = mathop
        self._n_lanes = n_lanes

    @property
    def n_lanes(self) :
        return self._n_lanes

    @property
    def mathop(self) :
        return self._mathop

    def __getitem__(self, key) :
        qubits = self._qubits()
        if qubits is None :
            return None
        return qubits.get_states(self._mathop, key)


class Qubits :
    def __init__(self, pkg, processor, dtype) :
        self.pkg = pkg
        self.processor = processor
        self.dtype = dtype
        self.qreg_ordering = list()
        self.lanes = lanes.Lanes()
        self.qstates_list = list()

    def __del__(self) :
        self.qstates_list = None

    def reset(self) :
        self.lanes.clear()
        for qstates in self.qstates_list :
            del qstates
        self.qstates_list = []

    def get_n_lanes(self) :
        return len(self.lanes)


    @property
    def states(self) :
        return StateGetter(self, null, self.get_n_lanes())

    @property
    def prob(self) :
        return StateGetter(self, abs2, self.get_n_lanes())
    
    def get_qubit_states_list(self) :
        return self.qstates_list

    def update_external_layout(self) :
        # sync external lane layout in self.lanes and self.qreg_ordering.
        for external_lane, qreg in enumerate(self.qreg_ordering) :
            lane = self.lanes.get(qreg)
            lane.set_external(external_lane)

    def _allocate_qubit_states(self, n_lanes) :
        # allocate qubit states
        qstates = self.pkg.create_qubit_states(self.dtype)
        self.processor.initialize_qubit_states(qstates, n_lanes)
        self.qstates_list.append(qstates)
        return qstates

    def add_qubit_states(self, qregset) :
        
        # initialize qubit states
        assert len(qregset) != 0, "empty qreg set."

        n_lanes = len(qregset)
        qstates = self._allocate_qubit_states(n_lanes)
        self.processor.reset_qubit_states(qstates)

        # sort qregset by qreg.id before lane asssignment.
        sorted_qreglist = sorted(qregset, key = lambda qreg:qreg.id)
        # create lane map.
        for local_lane, qreg in enumerate(sorted_qreglist) :
            self.lanes.add_lane(qreg, qstates, local_lane)
        self.qreg_ordering += sorted_qreglist

    def join(self, qregset) :

        # initialize qubit states
        assert 1 < len(qregset), "2 or more qregs required."

        # creating map, key qstates, value lane list.
        qstatesmap = dict()
        new_qregs = set()
        for qreg in qregset :
            if qreg in self.lanes :
                lane = self.lanes[qreg]
                lanelist = qstatesmap.get(lane.qstates, None)
                if lanelist is None :
                    lanelist = list()
                    qstatesmap[lane.qstates] = lanelist
                lanelist.append(lane)
            else :
                new_qregs.add(qreg)

        assert len(qstatesmap) <= len(self.qstates_list)

        # no exisiting qstates. create new one.
        if len(qstatesmap) == 0 :
            self.add_qubit_states(new_qregs)
            return

        # aggregate qubit states
        n_lanes = len(qregset)
        # sort by size, ascending order.
        qslist = sorted(qstatesmap.keys(), key = lambda qstates: qstates.get_n_lanes())

        # remove given qubit states
        for qs in qslist :
            self.qstates_list.remove(qs)
        # allocate qubit states
        joined = self._allocate_qubit_states(n_lanes)
        # join states in list
        self.processor.join(joined, qslist, len(new_qregs))

        # update local lane.  The last qstates has lowest local lanes.
        lane_offset = 0
        for qs in reversed(qslist) :
            lanes = qstatesmap[qs]
            for lane in lanes :
                # copy lane state
                new_local_lane = lane.local + lane_offset
                joined.set_lane_state(new_local_lane, qs.get_lane_state(lane.local))
                # update lane.  Should be done after copy.
                lane.update(joined, new_local_lane)

            lane_offset += len(lanes)

        # sort qregset by qreg.id before lane asssignment.
        sorted_qreglist = sorted(new_qregs, key = lambda qreg:qreg.id)
        # update qreg_order
        self.qreg_ordering += sorted_qreglist

        for new_qreg in sorted_qreglist :
            # update lane map.
            for idx, new_qreg in enumerate(sorted_qreglist) :
                local_lane = lane_offset + idx
                self.lanes.add_lane(new_qreg, joined, local_lane)

    def decohere_and_separate(self, qreg, value, prob) :
        sep_lane = self.lanes[qreg]
        qstates, local_lane = sep_lane.qstates, sep_lane.local  # qstates and lane to be separated.
        # allocate qubit states
        n_lanes = qstates.get_n_lanes()

        qstates0 = self._allocate_qubit_states(n_lanes - 1)        
        qstates1 = self._allocate_qubit_states(1)
        # update qstates_list.
        self.qstates_list.remove(qstates)

        # separate.  lane states is updated in processor.separate().
        self.processor.decohere_and_separate(value, prob, qstates0, qstates1, qstates, local_lane)
        # update lane states
        lanes = self.lanes.get_by_qubit_states(qstates)
        lanes.remove(sep_lane)  # remove target lane from the loop.
        for lane in lanes :
            new_local_lane = lane.local
            if local_lane < new_local_lane :
                new_local_lane -= 1
            qstates0.set_lane_state(new_local_lane, qstates.get_lane_state(lane.local))
        qstates1.set_lane_state(0, qstates.get_lane_state(local_lane))

        # update lanes
        for lane in lanes :
            lane.qstates = qstates0
            if sep_lane.local < lane.local :
                lane.local -= 1
        sep_lane.qstates, sep_lane.local = qstates1, 0
        # FIXME: add consistency checks.

    def deallocate_qubit_states(self, qreg) :
        lane = self.lanes[qreg]
        if lane.qstates.get_n_lanes() != 1 :
            raise RuntimeError('qreg/lane is not separated.')
        if lane.qstates.get_lane_state(0) == -1 :
            raise RuntimeError('qreg/lane is not measured.')
        # removing qreg from self.qreg_order.
        self.qreg_ordering.remove(qreg)
        # update lanes
        self.lanes.pop(qreg)
        # update qstates_list
        self.qstates_list.remove(lane.qstates)
        # release qubit states
        lane.qstates = None
        
    def calc_probability(self, qreg) :
        from qgate.model import Qreg
        if not isinstance(qreg, Qreg) :
            raise RuntimeError('qreg must be an instance of class Qreg.')
        
        lane = self.lanes[qreg]
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
