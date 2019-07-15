Measurement and Probability
===========================

Qgate implements single- and multiple-qubit measurements.  Measurements returns 0 or 1 as measured values.

Qgate also supports probability calculation for a specified qubits(qregs).  It returns propability for qregs as a floating point number.


Single-bit measurement
----------------------

measure(ref, qreg)

measure(ref, qreg) measures a qreg, and store a measured value which is referenced by ref.


.. code-block:: python
		
   qreg = new_qreg()
   # ... applying gates to qreg
   
   ref = new_reference()
   m = measure(ref, qreg)
   circuit.append(m)  # adding measure operation


Multiple-qubit measurement
--------------------------

measure(ref, gatelist)

measure(ref, gatelist) measures multiple-qregs, and observable is defined by gatelist.  A measurement result is stored in a value referenced by ref.


.. code-block:: python
		
   qreg = new_qreg()
   # ... applying gates to qreg
   
   ref = new_reference()
   gatelist = [X(qreg0), Y[qreg1), Z(qreg2)]
   m = measure(ref, gatelist)
   circuit.append(m)  # adding multi-qubit measure operation


   # single-bit measurement is identical to the following.
   m = measure(ref, Z(qreg)
   circuit.append(m)


Single-bit probability
----------------------

prob(ref, qreg)

prob(ref, qreg) calculates probability for a specified qreg.

and store a measured value which is referenced by ref.


.. code-block:: python
		
   qreg = new_qreg()
   # ... applying gates to qreg
   
   ref = new_reference()
   p = prob(ref, qreg)
   circuit.append(p)  # adding prob operation



Multiple-qubit probability
--------------------------

prob(ref, gatelist)

prob(ref, gatelist) calculates probability measures multiple-qregs, and observable is defined by gatelist.  A measurement result is stored in a value referenced by ref.


.. code-block:: python
		
   qreg = new_qreg()
   # ... applying gates to qreg
   
   ref = new_reference()
   gatelist = [X(qreg0), Y[qreg1), Z(qreg2)]
   p = prob(ref, gatelist)
   circuit.append(p)  # adding multi-qubit prob operation
