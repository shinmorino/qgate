Measurement and Probability
===========================

Qgate implements measure and prob operators.  Measure operator is to make a measurement, and prob operator calculates probability for specified qreg(s).

single- and multiple-qubit versions are provided for both measurement and prob operators.


Measurement
^^^^^^^^^^^

Single-bit measurement
----------------------

measure(ref, qreg)

measure(ref, qreg) measures a qreg, and store result in a value referred by ref.

Measure operation has a reference as the first parameter.  This reference is used to access to measured values after simulation.  The second parameter is qreg, on which measurement is performed.

Single qubit measuremenet is Z-based, and identical to measurement operation defined in OpenQASM.

.. code-block:: python

   ref = new_reference()
   m = measure(ref, qreg)
   circuit.append(m)


Multiple-qubit measurement
--------------------------

measure(ref, gatelist)

measure(ref, gatelist) is the operator for multiple-qubit measurement which is an equivalent of `Pauli Measurements in Q# <https://docs.microsoft.com/en-us/quantum/concepts/pauli-measurements>`_.

gatelist is a sequence of pauli gates to specify obervation operator.

A measurement result is stored in a value referenced by ref.

The first parameter is a reference, which is used to store the measured value.  The second parameter is a sequence of pauli and identity gates.

.. code-block:: python

   ref = new_reference()
   gatelist = [X(qreg0), Y(qreg1)]
   m = measure(ref, gatelist)
   circuit.append(m)

   # single-bit measurement is identical to the following.
   m = measure(ref, Z(qreg))
   circuit.append(m)


Probability
^^^^^^^^^^^

Single-bit probability
----------------------

prob(ref, qreg)

prob(ref, qreg) calculate probablity, :math:`Pr(Zero||\psi>)`, on a specified qreg.  Calculated provabity is stored to a value referred by ref.

.. code-block:: python
   
   ref = new_reference()
   p = prob(ref, qreg)
   circuit.append(p)  # adding prob operation


Multiple-qubit probability
--------------------------

prob(ref, gatelist)

prob(ref, gatelist) calculates probability on multiple-qregs, and observable is defined by gatelist.  Calculated probability is stored in a value referenced by ref.

.. code-block:: python

   ref = new_reference()
   gatelist = [X(qreg0), Y(qreg1)]
   m = prob(ref, gatelist)
   circuit.append(m)

   # single-bit measurement is identical to the following.
   m = prob(ref, Z(qreg))
   circuit.append(m)
