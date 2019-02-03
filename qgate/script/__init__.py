from __future__ import absolute_import

from .script import allocate_qreg, allocate_qregs, allocate_creg, allocate_cregs, new_circuit, measure, barrier, reset, clause, if_
# gate factory
from .script import a, h, s, t, x, y, z, rx, ry, rz, u1, u2, u3, controlled, cntr

#from . import qelib1
from qgate.model.processor import process
