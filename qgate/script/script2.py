from __future__ import absolute_import

from .script import new_qreg, new_qregs, release_qreg, new_reference, new_references, new_gatelist, measure, prob, barrier, reset, if_
# gate factory
from .script import I, H, S, T, X, Y, Z, Rx, Ry, Rz, U1, U2, U3, controlled, ctrl, Swap, SH, Expii, Expiz, Expi


ReleaseQreg = release_qreg
Measure = measure
Prob = prob
Barrier = barrier
Reset = reset
If = if_

del release_qreg
del measure
del prob
del barrier
del reset
del if_

# gate factory
Controlled = controlled
Ctrl = ctrl

del controlled
del ctrl
