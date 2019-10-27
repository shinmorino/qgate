# This example measures phase value in single qubit.

# importing qgate packages
import qgate
from qgate.script import *
import math

def qft_internal_layer(qregs) :
    ops = list()
    target = qregs[0]
    ops.append(H(target))
    for idx, ctrlbit in enumerate(qregs[1:]):
        theta = math.pi / float(1 << (idx + 1))
        r = ctrl(ctrlbit).U1(theta)(target)
        ops.append(r)
    return ops

# float value: 0.j1j2j3...jn
# qubits |j0j1j2 ... jn-1> -> qreg list [q0, q1, q2, ... , q_n-1)]
def qft(qregs) :
    ops = list()
    n_qregs = len(qregs)
    for idx in range(n_qregs):
        ops += qft_internal_layer(qregs[idx:])
    return ops

# float value: 0.j1j2j3...jn
# qubits |j0j1j2 ... jn-1> -> qreg list [q0, q1, q2, ... , q_n-1)]
def iqft(qregs) :
    ops = qft(qregs)
    ops.reverse()
    for op in ops:
        op.set_adjoint(True)
    return ops

def to_real(bits, n_bits):
    value = 0.
    for idx in range(n_bits):
        mask = 1 << idx
        if bits & mask != 0:
            value += math.pow(0.5, idx + 1)
    return value

# value to be measured.
v_in = 0.1

# target qreg.
target = new_qreg()

# number of qregs for estimated phase value.
n_bits = 20
# qregs for iqft.
bits = new_qregs(n_bits)

# initialize
ops = [H(qreg) for qreg in bits]

# set phase in the target qreg.
# Ui = Expii(pi * theta * 2^i), U1, U2, U4, U8 ...
for idx, ctrlreg in enumerate(bits):
    theta = 2 * math.pi * v_in * (1 << idx)
    ops.append(ctrl(ctrlreg).Expii(theta)(target))

# iqft gate sequence
ops += iqft(bits)
# dump circuit
qgate.dump(ops)

# run simulator
sim = qgate.simulator.cpu(prep_opt='dynamic')
sim.run(ops)

#sim.qubits.set_ordering(reversed(bits))
#qgate.dump(sim.qubits.prob)

# creating sampling pool
pool = sim.qubits.create_sampling_pool(bits)
# sample 1024 times.
obs = pool.sample(1024)
# creating histgram
hist = obs.histgram()
#print(hist)

# converting sampled values to real number.
results = list()
for ival, occurence in hist.items():
    fval = to_real(ival, n_bits)
    results.append((fval, occurence))
results.sort(key = lambda r:r[0])
for r in results:
    print(r)

import matplotlib.pyplot as plt

diff = [r[0] - v_in for r in results]
height = [r[1] for r in results]
plt.bar(diff, height, width = 0.8 * math.pow(0.5, n_bits))
x_delta = 25 * math.pow(0.5, n_bits)
plt.xlim(- x_delta, x_delta)
plt.xlabel('difference')
plt.ylabel('counts')
plt.show()
