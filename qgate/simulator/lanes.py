

class Lane :
    def __init__(self, qstates, local) :
        self.update(qstates, local)

    def update(self, qstates, local) :
        self.qstates = qstates
        self.local = local

    def set_external(self, external) :
        self.external = external


class Lanes(dict) :
    def __init__(self) :
        super(dict, self).__init__()

    def add_lane(self, qreg, qstates, local) :
        self[qreg] = Lane(qstates, local)

    def get_by_qubit_states(self, qs) :
        lanes = list()
        for qreg, lane in self.items() :
            if lane.qstates == qs :
                lanes.append(lane)
        return lanes

    # FIXME: improve interface.
    def get_state_index(self, *qregs) :
        idx = 0
        for qreg in qregs :
            idx |= 1 << self[qreg].external
        return idx
