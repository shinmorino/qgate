import numpy as np
from . import lanes
from . import empty_sampling_pool

# math operations
def null(v) :
    return v
def abs2(v) :
    return v.real ** 2 + v.imag ** 2
prob = abs2

        
# FIXME: implement __iter__ method.
class StateGetter :
    def __init__(self, qubits, mathop, n_qregs) :
        import weakref
        self._qubits = weakref.ref(qubits)
        self._mathop = mathop
        self._n_qregs = n_qregs

    @property
    def n_qregs(self) :
        return self._n_qregs

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
        self.states_getter = pkg.create_qubits_states_getter(dtype)
        self.dtype = dtype
        self._given_ordering = None
        self._ordering = list()
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

    def get_n_qregs(self) :
        qregset = set(self.lanes.keys())
        if self._given_ordering is not None :
            qregset |= set(self._given_ordering)
        return len(qregset)

    @property
    def ordering(self) :
        return self._ordering

    def set_ordering(self, qreglist) :
        self._given_ordering = qreglist
        self.update_external_layout()

    @property
    def states(self) :
        return StateGetter(self, null, self.get_n_qregs())

    @property
    def prob(self) :
        return StateGetter(self, abs2, self.get_n_qregs())
    
    def get_qubit_states_list(self) :
        return self.qstates_list

    def update_external_layout(self) :
        # sync external lane layout in self.lanes and self.qreg_ordering.
        qregset = set(self.lanes.keys())
        qreglist = list()
        if self._given_ordering is not None :
            qreglist += self._given_ordering
            qregset -= set(qreglist)
        # sort remaining qregs by qreg.id.
        remaining = sorted(list(qregset), key = lambda qreg:qreg.id)
        # add remainings
        self._ordering = qreglist + remaining

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

        # create lane map.
        for local_lane, qreg in enumerate(qregset) :
            self.lanes.add_lane(qreg, qstates, local_lane)

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

        # update lane map.
        for idx, new_qreg in enumerate(new_qregs) :
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

    def create_sampling_pool(self, qreg_ordering, sampling_pool_factory = None) :
        if len(set(qreg_ordering)) != len(qreg_ordering) :
            raise RuntimeError('qreg_ordering has duplicate qregs, {}.'.format(repr(qreg_ordering)))

        pool_ordering = list()
        empty_lane_qreglist = list()
        for qreg in qreg_ordering :
            if qreg in self.lanes :
                pool_ordering.append(qreg)
            else :
                empty_lane_qreglist.append(qreg)

        n_states = 1 << len(pool_ordering)
        if n_states == 1 :  # FIXME: ...
            return empty_sampling_pool.EmptySamplingPool(qreg_ordering)

        lane_trans = lanes.create_lane_transformation(self.lanes, pool_ordering)
        empty_lanes = list()
        for qreg in empty_lane_qreglist :
            empty_lanes.append(qreg_ordering.index(qreg))

        n_pool_lanes = len(pool_ordering)
        n_hidden_lanes = len(self.lanes) - len(pool_ordering)

        return self.states_getter.create_sampling_pool(qreg_ordering,
                                        n_pool_lanes, n_hidden_lanes, lane_trans, empty_lanes,
                                        sampling_pool_factory)

    def get_states(self, mathop = null, key = None) :
        if mathop == null :
            dtype = np.complex64 if self.dtype == np.float32 else np.complex128
        elif mathop == abs2 :
            dtype = self.dtype
        else :
            raise RuntimeError('unsupported mathop, {}'.format(repr(mathop)))

        # creating empty lane pos list.
        empty_lanes = list()
        if self._given_ordering is not None :
            for external_lane, qreg in enumerate(self._given_ordering) :
                if not qreg in self.lanes :
                    empty_lanes.append(external_lane)

        lane_trans = lanes.create_lane_transformation(self.lanes, self._ordering)
            
        n_states = 1 << self.get_n_qregs()
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
                return np.ones([0], dtype)
            
            values = np.empty([n_states], dtype)
            self.states_getter.get_states(values, 0, mathop,
                                          lane_trans, empty_lanes,
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
        self.states_getter.get_states(values, 0, mathop,
                                      lane_trans, empty_lanes,
                                      1, idx, 1)

        return values[0]
