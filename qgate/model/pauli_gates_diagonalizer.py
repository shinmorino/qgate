from . import model
from . import gate_type as gtype
from .gate_factory import *
import math

def _isID(op) :
    return isinstance(op.gate_type, gtype.ID)

class PauliGatesDiagonalizer :
    def __init__(self, gatelist) :
        self.gatelist = gatelist

    def expand_in_lane(self, gates, qreg) :
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

        updated_n_gates = len(xzlist)
        n_gates = 0
        while n_gates != updated_n_gates :
            pos = xzlist.find('XX')
            if pos != -1 :
                xzlist = xzlist[:pos] + xzlist[pos + 2:]
            pos = xzlist.find('ZZ')
            if pos != -1 :
                xzlist = xzlist[:pos] + xzlist[pos + 2:]
            n_gates = updated_n_gates
            updated_n_gates = len(xzlist)

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
                # s.h.(i * z).h.s+
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
                # s.h.(-i * z).h.s+
                sign += 3
                d, p = True, sh(qreg)
            else :
                # x.z.x =  -1  0
                #           0  1
                # (-1 * z)
                sign += 2
                d, p = True, None

        return sign, d, p
            

    def diagonalize(self) :
        
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
        gmapitems = list(gmap.items())
        gmapitems.sort(key = lambda item:item[0].id)
        for qreg, glist in gmapitems :
            _sign, d, p = self.expand_in_lane(glist, qreg)
            sign += _sign
            if d : # d is True when z-based.
                paulis.append(qreg)
                if p is not None :
                    self.plist.append(p)
            else :
                ids.append(qreg)

        self.phase_offset_in_pi_2 = sign % 4
            
        self.cxchain = []
        if len(paulis) == 0 :
            self.op_qreg = ids[0]
            return False # I-based
        
        for i in range(len(paulis) - 1) :
            d0, d1 = paulis[i], paulis[i + 1]
            self.cxchain.append(cx(d0, d1))
        self.op_qreg = paulis[-1]

        return True # Z-based

    def get_phase_offset_in_pi_2(self) :
        return self.phase_offset_in_pi_2

    def get_phase_coef(self) :
        phase_offset = self.phase_offset_in_pi_2 * math.pi / 2.
        return math.cos(phase_offset) + 1.j * math.sin(phase_offset)

    def get_pcx(self) :
        pcx = self.plist + self.cxchain
        return [op.copy() for op in pcx]
