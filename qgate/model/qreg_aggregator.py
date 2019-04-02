from . import model
from .pseudo_operator import ClauseBegin, ClauseEnd


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
        print('qreg {}'.format(qreg.id))
        for qregset in self.qregsetlist :
            ids = ['{}'.format(qreg.id) for qreg in qregset]
            print(' '.join(ids))
        assert False, 'Must not reach.'
        
    def add_qreg(self, qreg) :
        if not qreg in self.qregset :
            self.qregset.add(qreg)
            self.qregsetlist.append({qreg})
            return True
        return False
            
    def aggregate(self, qreglist) :
        qridset = set()
        for qreg in qreglist :
            qregset = self._find_qregset(qreg)
            qridset.add(id(qregset))
        if len(qridset) == 1 :
            return None
        
        merged = set()
        for qreg in qreglist :
            qregset = self._find_qregset(qreg)
            self.qregsetlist.remove(qregset)
            merged |= qregset
        self.qregsetlist.append(merged)

        return merged

