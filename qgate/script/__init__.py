from __future__ import absolute_import

from .script import allocate_qreg, allocate_qregs, allocate_creg, allocate_cregs, new_circuit, measure, barrier, reset, clause, if_
from . import qelib1
from qgate.model.processor import process
