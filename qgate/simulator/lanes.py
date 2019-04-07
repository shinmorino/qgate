

class Lane :
    def __init__(self, external, qstates, local) :
        self.update(external, qstates, local)

    def update(self, external, qstates, local) :
        self.external = external
        self.qstates = qstates
        self.local = local


class Lanes :
    def __init__(self) :
        self.reset()

    def reset(self) :
        self.lanes = dict()

    def exists(self, qreg) :
        return qreg.id in self.lanes.keys()

    def add_lane(self, qreg, external, qstates, local) :
        self.lanes[qreg.id] = Lane(external, qstates, local)

    def remove(self, qreg) :
        return self.lanes.pop(qreg.id)

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
            if qreg.id in self.lanes.keys() :
                lane = self.get(qreg)
                lanelist.append(lane)
            else :
                qreglist.add(qreg)
        return lanelist, qreglist

    def all(self) :
        return self.lanes.values()
