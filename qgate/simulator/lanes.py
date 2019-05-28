

class Lane :
    def __init__(self, qstates, local) :
        self.update(qstates, local)

    def update(self, qstates, local) :
        self.qstates = qstates
        self.local = local

    def set_external(self, external) :
        self.external = external


class Lanes :
    def __init__(self) :
        self.reset()

    def reset(self) :
        self.lanes = dict()

    def exists(self, qreg) :
        return qreg.id in self.lanes

    def add_lane(self, qreg, qstates, local) :
        self.lanes[qreg.id] = Lane(qstates, local)

    def remove(self, qreg) :
        removed = self.lanes.pop(qreg.id)

    def get_n_lanes(self) :
        return len(self.lanes)

    def __getitem__(self, id) :
        return self.lanes[id]

    def get(self, qreg) :
        return self.lanes[qreg.id]

    def get_by_id(self, id) :
        return self.lanes[id]

    def get_by_qubit_states(self, qs) :
        lanes = list()
        for qreg, lane in self.lanes.items() :
            if lane.qstates == qs :
                lanes.append(lane)
        return lanes

    def get_state_index(self, *qregs) :
        idx = 0
        for qreg in qregs :
            idx |= 1 << self.lanes[qreg.id].external
        return idx

    def get_lanes(self, qregset) :
        lanelist = []
        qreglist = []
        for qreg in qregset :
            if qreg.id in self.lanes :
                lane = self.get(qreg)
                lanelist.append(lane)
            else :
                qreglist.add(qreg)
        return lanelist, qreglist

    def all(self) :
        return self.lanes.values()
