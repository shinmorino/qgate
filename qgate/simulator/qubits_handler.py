
class QubitsHandler :
    def __init__(self, qubit_states_factory, qubits) :
        self._create_qubit_states = qubit_states_factory
        self._qubits = qubits

    def reset(self) :
        self._qubits.reset()

    @property
    def lanes(self) :
        return self._qubits.lanes
    
    def _allocate_qubit_states(self, n_lanes) :
        # allocate qubit states
        qstates = self._create_qubit_states(self._qubits.dtype)
        qstates.processor.initialize_qubit_states(qstates, n_lanes)
        self._qubits.qstates_list.append(qstates)
        return qstates

    def add_qubit_states(self, qregset) :
        
        # initialize qubit states
        assert len(qregset) != 0, "empty qreg set."

        n_lanes = len(qregset)
        qstates = self._allocate_qubit_states(n_lanes)
        qstates.processor.reset_qubit_states(qstates)

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

        assert len(qstatesmap) <= len(self._qubits.qstates_list)

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
            self._qubits.qstates_list.remove(qs)
        # allocate qubit states
        joined = self._allocate_qubit_states(n_lanes)
        # join states in list
        joined.processor.join(joined, qslist, len(new_qregs))

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
        self._qubits.qstates_list.remove(qstates)

        # separate.  lane states is updated in processor.separate().
        # FIXME: correct propcessor ?
        qstates.processor.decohere_and_separate(value, prob,
                                                qstates0, qstates1, qstates, local_lane)
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
        self._qubits.qstates_list.remove(lane.qstates)
        # release qubit states
        lane.qstates = None
