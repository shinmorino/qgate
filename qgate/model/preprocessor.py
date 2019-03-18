from . import model
from .pseudo_operator import FrameBegin, FrameEnd

class Preprocessor :
    def __init__(self) :
        self.reset()

    def reset(self) :
        self._qregsetlist = list()
        self._qregset = set()
        self._refset = set()

    def get_qregsetlist(self) :
        return self._qregsetlist

    def get_refset(self) :
        return self._refset
        
    def _add_qreg(self, qreg) :
        if not qreg in self._qregset :
            self._qregset.add(qreg)
            self._qregsetlist.append({qreg})

    def _add_ref(self, ref) :
        if not ref in self._refset :
            self._refset.add(ref)

    def _merge_qreglist(self, qreglist) :
        qridset = set()
        for qreg in qreglist :
            qregset = self._find_qregset(qreg)
            qridset.add(id(qregset))
        if len(qridset) != 1 :
            merged = set()
            for qreg in qreglist :
                qregset = self._find_qregset(qreg)
                self._qregsetlist.remove(qregset)
                merged |= qregset
            self._qregsetlist.append(merged)

    def _find_qregset(self, qreg) :
        for qregset in self._qregsetlist :
            if qreg in qregset :
                return qregset
        print('qreg {}'.format(qreg.id))
        for qregset in self._qregsetlist :
            ids = ['{}'.format(qreg.id) for qreg in qregset]
            print(''.join(ids))
        assert False, 'Must not reach.'
    
    def collect_qregs(self, clause) :
        for op in clause.ops :
            if isinstance(op, model.Measure) :
                self._add_qreg(op.qreg)
                self._add_ref(op.outref)
            elif isinstance(op, model.Pmeasure) :
                for gate in op.gatelist :
                    self._add_qreg(gate.qreglist[0])
            elif isinstance(op, model.Gate) :
                for qreg in op.qreglist :
                    self._add_qreg(qreg)
                if not op.ctrllist is None :
                    for qreg in op.ctrllist :
                        self._add_qreg(qreg)
                    self._merge_qreglist(op.qreglist + op.ctrllist)
            elif isinstance(op, (model.Barrier, model.Reset)) :
                for qreg in op.qregset :
                    self._add_qreg(qreg)
            elif isinstance(op, model.Clause) :
                self.collect_qregs(op)
            elif isinstance(op, model.IfClause) :
                self.collect_qregs(op.clause)
            elif isinstance(op, (FrameBegin, FrameEnd)) :
                pass
            else :
                raise RuntimeError(repr(op))

    def _get_qregset_idx(self, qreg) :
        for idx, qrset in enumerate(self._qregsetlist) :
            if qreg in qrset :
                return idx
        assert False, 'Must not reach'
    
    # clone operators if an operator belongs two or more circuits
    def label_qregset_idx(self, clause) :
        for op in clause.ops :
            if isinstance(op, model.Measure) :
                op.qregset_idx = self._get_qregset_idx(op.qreg)
            elif isinstance(op, model.Pmeasure) :
                # qregset_idx must be the same for all child gates
                op.qregset_idx = self._get_qregset_idx(op.gatelist[0].qreglist[0])
            elif isinstance(op, model.Gate) :
                # normal gate
                assert len(op.qreglist) == 1
                op.qregset_idx = self._get_qregset_idx(op.qreglist[0])
                if op.ctrllist is not None : # FIXME: remove after debug
                    assert all([op.qregset_idx == self._get_qregset_idx(ctrlreg)
                                for ctrlreg in op.ctrllist])
            elif isinstance(op, (model.Barrier, model.Reset)) :
                assert(len(op.qregset) == 1)
                op.qregset_idx = self._get_qregset_idx(*op.qregset)
            elif isinstance(op, model.IfClause) :
                # FIXME: add barriers, op.qregset_idx required ?
                pass
            elif isinstance(op, (FrameBegin, FrameEnd)):
                pass
            elif isinstance(op, model.Clause) :
                assert False, 'Clause must not appear in processor.'
            else :
                assert False, 'Unknown operator'

    def preprocess(self, clause) :
        self.collect_qregs(clause)
        
        # give operators their numbers.
        for idx, op in enumerate(clause.ops) :
            op.set_idx(idx)
        self.label_qregset_idx(clause)
