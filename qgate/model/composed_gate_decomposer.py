import qgate.model as model
import qgate.model.gate_type as gtype
from .gate_factory import *

def _isID(op) :
    return isinstance(op.gate_type, gtype.ID)

class ComposedGateDecomposer :
    def __init__(self, composed) :
        self.composed = composed

    def filter_paulis(self) :
        # id gates are ignored.
        self.paulis = []
        for gate in self.composed.gatelist :
            if not _isID(gate) : # extract pauli gates
                self.paulis.append(gate)
        return len(self.paulis) != 0

    def decompose(self) :
        # collect gates according to qregs.
        gmap = dict()
        for gate in self.paulis :
            qreg = gate.qreglist[0]
            glist = gmap.get(qreg, None)
            if glist is None :
               gmap[qreg] = [gate]
            else :
                glist.append(gate)

        glist_even, glist_odd = [], []
        # decompose gmap for qregs 
        for glist in gmap.values() :
            if len(glist) % 2 == 0 :
                glist_even += glist
            else :
                glist_odd += glist

        self.plist = []
        for gate in glist_even + glist_odd :
            if isinstance(gate.gate_type, gtype.X) :
                # X = H Z H
                self.plist.append(h(gate.qreglist))
            elif isinstance(gate.gate_type, gtype.Y) :
                # Y = (SH) Z (HS+)
                self.plist.append(sh(gate.qreglist))
            elif isinstance(gate.gate_type, gtype.Z) :
                pass
            else :
                raise RuntimeError('input must be X, Y, Z or ID, but {} passed.'.format(gate.gate_type))

        self.cxlist = []
        for i in range(len(glist_even) - 1) :
            d0, d1 = glist_even[i].qreglist[0], glist_even[i + 1].qreglist[0]
            self.cxlist.append(ca(d0, d1))

        if len(glist_odd) == 0 :
            self.is_z_based = False
            self.op_qreg = glist_even[-1].qreglist[0]
        else :
            if len(glist_even) != 0 :
                d0, d1 = glist_even[-1].qreglist[0], glist_odd[0].qreglist[0]
                self.cxlist.append(ca(d0, d1))
            for i in range(len(glist_odd) - 1) :
                d0, d1 = glist_odd[i].qreglist[0], glist_odd[i + 1].qreglist[0]
                self.cxlist.append(cx(d0, d1))
            self.op_qreg = glist_odd[-1].qreglist[0]
            self.is_z_based = True

        return self.is_z_based
            

    def get_pcx(self) :
        cxlist = self.cxlist if not self.composed.adjoint else reversed(self.cxlist)
        pcx = self.plist + cxlist
        if self.composed.ctrllist is not None :
            for gate in pcx :
                gate.set_ctrllist(list(self.composed.ctrllist))
        return pcx

    def get_pcxdg(self) :
        cxlist = reversed(self.cxlist) if not self.composed.adjoint else self.cxlist
        pdglist = [p.copy() for p in reversed(self.plist)]
        # create adjoint P
        for pdg in pdglist :
            pdg.set_adjoint(True)
        pcxdg = list(cxlist) + pdglist
        if self.composed.ctrllist is not None :
            for gate in pcxdg :
                gate.set_ctrllist(list(self.composed.ctrllist))
        return pcxdg
