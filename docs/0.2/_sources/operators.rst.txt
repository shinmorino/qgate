Operators
=========

In qgate, quantum circuits are defined as sequences of operators.  Please refer below for operators provided in Qgate.
  
  * :ref:`gate:Gate`
    
  * :ref:`measurement:Measurement`
    
  * :ref:`measurement:Probability`
    
  * :ref:`if_clause:If clause`
    
  * Reset_
    
  * Barrier_


Reset
^^^^^

Reset operation resets one or multiple qubits to `|0>` state.

A qreg should be measured before Reset operation.  The qubits is not measured, Reset will raise an error.

.. code-block:: python

   # resetting qubit
   circuit.append(reset(qreg))

   # equivalent code.
   
   measure(ref, qreg)     # qreg is measured somewhere before reset().
   ...
   if_(ref, 1, X(qreg))   # negate qreg when a measured value is 1.
   ...
   reset(qreg)            # resetting qreg to |0>
   
   
Barrier
^^^^^^^

Barrier operation works as barrier on quantum circuit optimization.  Barrier operator can accept single or multiple qregs.

.. code-block:: python

   # barrier, 1 qreg
   circuit.append(barrier(qreg))

   # barrier, 2 qregs
   circuit.append(barrier([qreg0, qreg1]))

