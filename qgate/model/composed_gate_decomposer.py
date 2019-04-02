from . import model
from . import gate_type as gtype
from .gate_factory import *
import math

def _isID(op) :
    return isinstance(op.gate_type, gtype.ID)

class ComposedGateDecomposer :
    def __init__(self, gatelist) :
        self.gatelist = gatelist

    def compose_gates(self, gates, qreg) :
        sign = 0
        xzlist = ''
        
        for gate in gates :
            if isinstance(gate.gate_type, gtype.X) :
                xzlist += 'X'
            elif isinstance(gate.gate_type, gtype.Y) :
                xzlist += 'XZ'
                sign += 1
            elif isinstance(gate.gate_type, gtype.Z) :
                xzlist += 'Z'
            elif isinstance(gate.gate_type, gtype.ID) :
                pass
            else :
                raise RuntimeError('input must be X, Y, Z or ID, but {} passed.'.format(gate.gate_type))

        while True :
            modified = False
            pos = xzlist.find('XX')
            if pos != -1 :
                xzlist = xzlist[:pos] + xzlist[pos + 2:]
                modified = True
            pos = xzlist.find('ZZ')
            if pos != -1 :
                xzlist = xzlist[:pos] + xzlist[pos + 2:]
                modified = True
            if not modified :
                break

        # xzlist is '(zx)*z?' if is_zx is True.
        # otherwise, '(xz)*x?'
        n_zx = len(xzlist) // 2
        has_trailing_gate = len(xzlist) % 2 != 0

        n_zx %= 4
        if 2 <= n_zx :
            sign += 2
            n_zx -= 2

        if n_zx == 0 and not has_trailing_gate :
            # empty, expiI()
            return sign, False, None
        
        head_is_z = xzlist[0] == 'Z'
        if n_zx == 0 : # single gate
            assert has_trailing_gate
            if head_is_z :
                d, p = True, None
            else :
                d, p = True, h(qreg)
            return sign, d, p

        assert n_zx == 1
        if head_is_z :
            if not has_trailing_gate :
                # z.x =  0 1
                #       -1 0
                # s.h.(i * z).h+.s
                sign += 1
                d, p = True, sh(qreg)
            else :
                # z.x.z =   0 -1
                #          -1  0
                # h.(-1 * z).h
                sign += 2
                d, p = True, h(qreg)
        else :
            if not has_trailing_gate :
                # x.z =  0 -1
                #        1  0
                # s.h.(-i * z).h+.s
                sign += 3
                d, p = True, sh(qreg)
            else :
                # x.z.x =  -1  0
                #           0  1
                # s.(-1 * z).s+
                sign += 2
                d, p = True, s(qreg)

        return sign, d, p
            

    def decompose(self) :
        
        # creating gate list for each qreg.
        gmap = dict()
        for gate in self.gatelist :
            glist = gmap.get(gate.qreg, None)
            if glist is None :
                glist = list()
                gmap[gate.qreg] = glist
            glist.append(gate)
            
        # compose pauli gates
        paulis, ids = list(), list()
        self.plist = list()
        sign = 0
        for qreg, glist in gmap.items() :
            _sign, d, p = self.compose_gates(glist, qreg)
            sign += _sign
            if d : # d is True when z-based.
                paulis.append(qreg)
                if p is not None :
                    self.plist.append(p)
            else :
                ids.append(qreg)

        sign = (sign % 4)
        if 3 <= sign :
            sign -= 4
        self.phase_offset = math.pi / 2. * sign
            
        self.ctrllist = []
        for i in range(len(ids) - 1) :
            d0, d1 = ids[i], ids[i + 1]
            self.ctrllist.append(ca(d0, d1))

        if len(paulis) == 0 :
            self.is_z_based = False
            self.op_qreg = ids[-1]
        else :
            if len(ids) != 0 :
                d0, d1 = ids[-1], paulis[0]
                self.ctrllist.append(ca(d0, d1))
            for i in range(len(paulis) - 1) :
                d0, d1 = paulis[i], paulis[i + 1]
                self.ctrllist.append(cx(d0, d1))
            self.op_qreg = paulis[-1]
            self.is_z_based = True

        return self.is_z_based

    def get_phase_offset(self) :
        return self.phase_offset

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
