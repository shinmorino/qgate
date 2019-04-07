
class QregAggregator :
    
    def __init__(self) :
        self.reset()

    def reset(self) :
        self.qregsetlist = list()
        self.qregset = set()

    def _find_qregset(self, qreg) :
        for qregset in self.qregsetlist :
            if qreg in qregset :
                return qregset
        # for debug
        print('qreg {}'.format(qreg.id))
        for qregset in self.qregsetlist :
            ids = ['{}'.format(qreg.id) for qreg in qregset]
            print(' '.join(ids))
        assert False, 'Must not reach.'
        
    def add_qreg(self, qreg) :
        if not qreg in self.qregset :
            self.qregset.add(qreg)
            self.qregsetlist.append(frozenset([qreg]))
            return True
        return False

    def aggregate(self, qreglist) :
        # add qreg if not added yet.
        for qreg in qreglist :
            if not qreg in self.qregset :
                self.add_qreg(qreg)
        # collect qregsets that contain qregs in the given qreglist
        qsset = set()
        for qreg in qreglist :
            qregset = self._find_qregset(qreg)
            qsset.add(qregset)
        # already aggregated.
        if len(qsset) == 1 :
            return None

        # merge qregset, and update qregsetlist.
        merged = set()
        for qs in qsset :
            self.qregsetlist.remove(qs)
            merged |= qs
        self.qregsetlist.append(frozenset(merged))
        
        return merged

    def separate_qreg(self, qreg) :
        # get qregset that contains the given qreg.
        qregset = self._find_qregset(qreg)
        # remove qregset from qregsetlist.
        self.qregsetlist.remove(qregset)
        # remove qreg from qregset.
        qregset = set(qregset)
        qregset.remove(qreg)
        # append 2 seperated qregsets.
        self.qregsetlist.append(frozenset(qregset))
        self.qregsetlist.append(frozenset([qreg]))

    # for debug
    def validate(self) :
        import numpy as np

        n_qregs_0 = np.sum([len(qregset) for qregset in self.qregsetlist])
        n_qregs_1 = len(self.qregset)
        assert n_qregs_0 == n_qregs_1

