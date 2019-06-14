

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
def create_lane_transformation(lanes, qreg_ordering) :
    qsdict = dict()
    for qreg, lane in lanes.items() :
        lanelist = qsdict.get(lane.qstates)
        if lanelist is None :
            lanelist = list()
            qsdict[lane.qstates] = lanelist
        lane = Lane(qreg, lane.local)
        if qreg in qreg_ordering :
            external_idx = qreg_ordering.index(qreg)
        else :
            external_idx = -1
        lane.set_external(external_idx)
        lanelist.append(lane)

    transformation = list()
    for qs, lanelist in qsdict.items() :
        lanelist.sort(key = lambda lane: lane.local)        
        transformation.append((qs, lanelist))
    transformation.sort(key = lambda tr: tr[1][0].local)

    return transformation
