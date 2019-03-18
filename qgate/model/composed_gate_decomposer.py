import qgate.model as model
import qgate.model.gate_type as gtype
from .gate_factory import *

def _isID(op) :
    return isinstance(op.gate_type, gtype.ID)

class ComposedGateDecomposer :
    def __init__(self, gatelist) :
        self.gatelist = gatelist

    def decompose(self) :
        # id gates are ignored.
        paulis, ids = [], []
        for gate in self.gatelist :
            if not _isID(gate) : # extract pauli gates
                paulis.append(gate)
            else :
                ids.append(gate)
        
        # collect gates according to qregs.
        gmap = dict()
        for gate in paulis :
            qreg = gate.qreglist[0]
            glist = gmap.get(qreg, None)
            if glist is None :
                glist = list()
                gmap[qreg] = glist
            glist.append(gate)

        # remove id gates when another pauli gate applied.
        ids_nodup = []
        for idgate in ids :
            if not idgate.qreglist[0] in gmap.keys() :
                ids_nodup.append(idgate)
        ids = ids_nodup
            
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

        self.ctrllist = []
        id_based = ids + glist_even
        for i in range(len(id_based) - 1) :
            d0, d1 = id_based[i].qreglist[0], id_based[i + 1].qreglist[0]
            self.ctrllist.append(ca(d0, d1))

        if len(glist_odd) == 0 :
            self.is_z_based = False
            self.op_qreg = id_based[-1].qreglist[0]
        else :
            if len(id_based) != 0 :
                d0, d1 = id_based[-1].qreglist[0], glist_odd[0].qreglist[0]
                self.ctrllist.append(ca(d0, d1))
            for i in range(len(glist_odd) - 1) :
                d0, d1 = glist_odd[i].qreglist[0], glist_odd[i + 1].qreglist[0]
                self.ctrllist.append(cx(d0, d1))
            self.op_qreg = glist_odd[-1].qreglist[0]
            self.is_z_based = True

        return self.is_z_based
            

    def get_pcx(self, adjoint) :
        ctrllist = self.ctrllist if not adjoint else reversed(self.ctrllist)
        pcx = self.plist + ctrllist
        return pcx

    def get_pcxdg(self, adjoint) :
        ctrllist = reversed(self.ctrllist) if not adjoint else self.ctrllist
        pdglist = [p.copy() for p in reversed(self.plist)]
        # create adjoint P
        for pdg in pdglist :
            pdg.set_adjoint(True)
        pcxdg = list(ctrllist) + pdglist
        return pcxdg
