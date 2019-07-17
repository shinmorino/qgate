Quick start guide
=================

This page covers followings:

#. `Simple example`_

#. `Code walkthrough`_
   
#. Operators_

#. `Running a circuit on a simulator`_

#. `Accessing simulation results`_

#. `Other examples`_


Simple example
--------------

The first example is to apply Hadamard gate to a qubit and measure it.

The original source is `here <https://github.com/shinmorino/qgate/blob/master/examples/simple_hadamard.py>`_ (`raw <https://raw.githubusercontent.com/shinmorino/qgate/master/examples/simple_hadamard.py>`_)

.. code-block:: python
   :linenos:
      
   # importing qgate packages
   import qgate
   from qgate.script import *
   
   # creating a qreg.
   qreg = new_qreg()

   # creating a quantum circuit with one H gate.
   circuit = [ H(qreg) ]

   # creating reference as a placeholder to store a result. 
   ref = new_reference()
   
   # adding measure operation
   m = measure(ref, qreg)
   circuit.append(m)

   # creating simulator instance, and run circuit.
   sim = qgate.simulator.cpu()
   sim.run(circuit)

   # accesing simulation results
   value = sim.values.get(ref)
   print('Measured value: ' + str(value))


Here is an output example:

.. code-block:: bash

   $ python simple_hadmard.py
   Measured value: 0

Meaured values can be 0 or 1.


Code walkthrough
^^^^^^^^^^^^^^^^

In this section, we'll have walkthrough the above sample.

.. code-block:: python
   :lineno-start: 1
   
   # importing qgate packages
   import qgate
   from qgate.script import *
   

The package name is **qgate**, that is imported before using qgate.

The **qgate.script** package provides factory functions of gates and other operators.  One needs to import this package to build quantum circuits.

Quantum circuits are built on circuit, and circuit is  used as an input of simulators.
Circuit holds a sequence of operators.

.. code-block:: python
   :lineno-start: 5
		  
   # creating a qreg
   qreg = new_qreg()

Creating a qreg.

Qreg is quantum register as deined in OpenQASM.  It is a logical representation of qubit.

The function of **new_qreg()** returns one qreg instance.  To create a list of qregs, use **new_qregs(n_qregs)**.
   

.. code-block:: python
   :lineno-start: 8

   # creating a quantum circuit with one H gate
   circuit = [ H(qreg) ]

A quantum circuit are defined as a sequence of Operators_, and python's list is used as a conatiner.  In this example, one Hadamard gate is added with a target bit specified by qreg.

Qgate implements various gates including controlled gates and adjoint. Please see `Gate`_ section for details.

.. code-block:: python
   :lineno-start: 11

   # creating reference as a placeholder to store a result. 
   ref = new_reference()

   # creating measure operation, and add it to the circuit.
   m = measure(ref, qreg)
   circuit.append(m)
   

For Measurement_, a reference is used as a placeholder of a measured value.
With a created reference, measure operation is appended to circuit.

.. code-block:: python
   :lineno-start: 18
   
   # creating simulator instance, and run circuit.
   sim = qgate.simulator.cpu()
   sim.run(circuit)


To run simulations, simulator instance is created by using **qgate.simulator.cpu()**.  Other simulator factory methods of **qgate.simulator.py()** and **qgate.simulator.cuda()** are available.

Simulator instance has run() method, which accepts a circuit as its parameter.  Simulator.run() returns after executing all operators in a given circuit.
   
.. code-block:: python
   :lineno-start: 22
		  
   # accesing simulation results
   value = sim.values.get(ref)
   print('Measured value: ' + str(value))

To get simulation results, **Simulator.values** property is used.  This property is a dictionary that holds measured values.

The 'ref' variable is used at line 15 to add measure operation, and the measured result is retrieved by using sim.values.get(ref).  Here, reference is used as a key to access to a value.  A resulting values of measurement is 0 or 1.

Probability_ operator is also available to get probability, :math:`Pr(Zero||\psi>)` on a specified qreg.  Also in this case, calculated probability is accessed in a similar way.


Operators
---------

In qgate, quantum circuits is defined as a sequence of operators, and qgate has folowing oeprators.
  
  * Gate_
    
  * Measurement_

  * Probability_
    
  * `If clause`_
    
  * Reset_
    
  * Barrier_


Gate
^^^^

Table 1 shows the list of built-in gates. Please visit `Builtin gate <gate.html>`_ for details.

================ ============================================================================
 Type            Gate
================ ============================================================================
 1 qubit gate     
   No parameter    ID, H, X, Y, Z, S, T
   1 parameter     Rx(theta), Ry(theta), Rz(theta), U1(_lambda), Expii(theta), Expiz(theta)
   2 parameters    U2(phi, _lambda)
   3 parameters    U3(theta, phi, lambda)
 Composed gate   Expi(theta)(gatelist)
 2 qubit gate    Swap(qreg0, qreg1)
================ ============================================================================

- Controlled gate

  All gates except for Swap gate works as controlled gate.  Applying multiple controlled bits is also supported.

- Adjoint

  All gates except for Swap gate have their adjoint.


Single qubit gate
+++++++++++++++++

The syntax shown shown is used to create single qubit gates.

Tokens surrounded by ``<>`` may appear 0- or 1-time according to a gate to be declared.

.. code-block:: python

  <ctrl(qregs).>GateType<(paramters)><.Adj>(qreg)

- Control bits

  ``ctrl(qregs).`` specify control bits.  It appears only when a gate has (a) control bit(s).
  A comma-separated list of qregs, a list of qregs, or their mixture is accepted.

- GateType<(parameters)>

  ``GateType`` is a gate name, such as H, Rx and Expii.
  If a specified gate type does not have any parameter, ``(paramters)`` is omitted.

- <.Adj>

  Specifying to use adjoint of a GateType.
  All gates except for Swap gate support adjoint.  Gate types that has hermite matrices such as H and X simply ignores ``.Adj``.

- (qreg)

  Qreg instance as a target qubit(qreg).


Examples:

.. code-block:: python

   # Hadamard gate
   H(qreg0)

   # Controlled X gate (CX gate)
   ctrl(qreg0).X(qreg1)

   # 2-control-bit X gate (Toffoli gate)
   ctrl(qreg0, qreg1).X(qreg2)

   # Rx gate (1 parameter)
   Rx(0.)(qreg)

   # Adjoint of Rx gate
   Rx(0.).Adj(qreg)

   # adjoint of 3-bit-controlled U3 gate
   # control bits are given by a python list.
   ctrlbits = [qreg0, qreg1, qreg2]  # creating a list of control bits
   ctrl(ctrlbits).U3(theta, phi, _lambda).Adj(qreg3)


Composed gate
+++++++++++++

Currently only 1 composed gate, Expi, is implemented.

The syntax to declare Expi gate is similar to other gates.  It allows to accept controll bits and supports adjoint.  But the operand is a list of pauli and ID gates.

.. code-block:: python

   <ctrl(qregs).>GateType<(paramters)><.Adj>(gatelist)

Examples:

.. code-block:: python

   # exp(i * math.pi * X), identical to Rx(math.pi).
   Expi(math.pi)(X(qreg))

   # can have a sequence of pauli operators
   Expi(math.pi / 2.)(X(qreg0), Y(qreg1), Z(qreg2))
   
   # Can be a controlled gate
   ctrl(qreg0).expi(math.pi)(Y(qreg1))
   
   # Supports adjoint
   expi(math.pi).Adj(Y(qreg1))
   

2 qubit gate
++++++++++++

Qgate implements Swap as a 2 qubit gate.

.. code-block:: python

   # Swap gate
   Swap(qreg0, qreg1)

   
   
Adding multiple gates easier
++++++++++++++++++++++++++++

Circuits are defined by using python's list.  So sequences of gates and operators can be created programatically.  Nested lists are allowed.

.. code-block:: python

   # example of nested list
   qregs = new_qregs(10)
   circuit = [
     [H(qreg) for qreg in qregs],  # creating a list with 10 H gates
     [X(qreg) for qreg in qregs]   # adding 10 X gates
   ]

   refs = new_references(10)
   # add 10 measure operators.
   circuit += [measure(ref, qreg) for ref, qreg in zip(refs, qregs)]


Measurement
^^^^^^^^^^^

Q gates implements 2 measure operations, (1) single-qubit measurement and (2) multi-qubit measurement.

Single qubit measurement
++++++++++++++++++++++++++++++++

Single qubit measuremenet is Z-based, and identical to measurement operation defined in OpenQASM.

Measure operation has a reference as the first parameter.  This reference is used to get measured values later.  The second parameter is qreg, on which measurement is performed.

.. code-block:: python

   ref = new_reference()
   m = measure(ref, qreg)
   circuit.append(m)


Multi-qubit measurement
+++++++++++++++++++++++

Multi qubit measurement has a sequence of pauli gates to specify obervation operator, which is an equivalent of `Pauli Measurements in Q# <https://docs.microsoft.com/en-us/quantum/concepts/pauli-measurements>`_.
The first parameter is a reference, and it's used to get measured values later.  The second parameter is a sequence of pauli and identity gates .

.. code-block:: python

   ref = new_reference()
   gatelist = [X(qreg0), Y(qreg1)]
   m = measure(ref, gatelist)
   circuit.append(m)

   
Probability
^^^^^^^^^^^

Probablity operators calculate probablity, :math:`Pr(Zero||\psi>)` on a specified qreg.

There're 2 probability operations, (1) single qubit probability and (2) multi-qubit probability.  Probability operators returns probability if measurement result is 0.  Single- and multi-qubit probability calculations are available.

.. code-block:: python

   ref = new_reference()
   
   # single qubit probability
   p = prob(ref, qreg)
   circuit.append(p)
   
   # multi qubit probability
   gatelist = [X(qreg0), Y(qreg1)]
   p = prob(ref, gatelist)
   circuit.append(p)


If clause
^^^^^^^^^

If clause is for conditional execution in quantum circuits.

if_(refs, cond, clause)

The first argument, refs, is one reference or a list of references.  The second parameter, cond, is a integer value or a function.  The third parameter, clause, is an operator or a list of operators.


cond as integer value
+++++++++++++++++++++

When ref is one reference, values referenced by refs are compared with the given integer value.  If they're equal, clause is executed.

The paramter, ref, can be a list of references.  In thie case, ref is converted to an integer value accodring to the code shown below, and compared with the cond value.  This mimics OpenQASM if statement.

If ref is not measured, referenced value is 0.

.. code-block:: python
		
   v = 0
   for idx, ref in enumerate(refs):
       if get_value_from_ref(ref) == 1 :
           v |= 1 << idx

   if v == cond :
       ... run operators in clause ...


Examples.

.. code-block:: python

   # if clause with one ref.
   circuit.append( if_(ref, 1, X(qreg)) )

   # if clause with 3 refs.
   refs = new_references(3)
   
   ... measure somewhere.

   # if values referred by refs[0] and refs[1] are 1,
   # and one referred by refs[2] is not 1,
   # X(qreg) is applied to qreg.
   circuit.append( if_(refs, 3, X(qreg)) )
   

cond as an function
+++++++++++++++++++

When refs is one reference, a value referenced by refs are passed to the function specified in cond.  If the function returns True, clause is executed.

If refs is a list of references, unpacked referred values are passed to the given function.  If the function returns True, clause is executed.

If ref is not measured, referenced value is None.

.. code-block:: python

   # if with one reference
   # if a value referred by ref is 0, X(qreg) is executed.
   i = if_(ref, lambda x: return x == 0, [X(qreg)])
   circuit.append(i)

   # if with a list of references
   refs = new_references(3)
   
   ... measure somewhere.

   # if values referred by refs[0] and refs[1] are 1, and a value referred by refs[2] is 0,
   # X(qreg) is applied to qreg.
   i = if_(refs,
           lambda v0, v1, v2 : return (v0 == 1) and (v1 == 1) and (v2 == 0),
	   X(qreg))
   circuit.append(i)


Reset
^^^^^

Reset operation resets one or multiple qubits to `|0>` state.

A qreg should be measured before Reset operation.

The qubits is not measured, Reset will raise an error.

.. code-block:: python

   # resetting qubit
   circuit.append(reset(qreg))

   # equivalent code.
   
   measure(ref, qreg)     # qreg is measured somewhere before reset().
   ...
   if_(ref, 1, X(qreg))   # negate qreg when a measured value is 1.
		
   
Barrier
^^^^^^^

Barrier operation works as barrier on quantum circuit optimization (to be implemented later versions).  Barrier can accept single or multiple qregs.

.. code-block:: python

   # barrier, 1 qreg
   circuit.append(barrier(qreg))

   # barrier, 2 qregs
   circuit.append(barrier([qreg0, qreg1]))



Running a circuit on a simulator
--------------------------------

Simulator instance is created by using **qgate.simulator.<runtime>()**, where runtime is py, cpu and cuda.

Qgate currently implements 3 versions of simulators, (1) python, (2) CPU(multicore), and (3) GPU(CUDA) versions, please visit `Simulator <simulator.html>`_ for details.

Simulator holds simulation results.  They are accessed from properties of **Simulator.values** and **Simulator.qubits**.

Accessing simulation results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Simulator.values** is a dictionary of values obtained during simulation.

In quantum circuits, references are used to create measure and prob operations.  By using references given for these opeartors, simulation results are retrieved.

Simulator.values.get() is used to get referred values, and it accepts one reference or a list of references.  If one reference is passed, one referred value is returned.  If a list of references is passed, a list of referred values is returned.

.. code-block:: python

   # getting one value associated with ref.
   v = sim.values.get(ref)

   # getting value list associated with reference list
   values = sim.values.get(refs)


Accessing state vector
^^^^^^^^^^^^^^^^^^^^^^

**Simulator.qubits.states** return a copy of state vector, and is accessed like numpy arrays. It acceepts slices.

**qgate.dump()** is avaialble to dump state vector.

.. code-block:: python

   # getting whole state vector.
   v = sim.qubits.states[:]

   # getting states for odd index-th elements.
   v = sim.qubits.states[1::2]

   # dump states
   qgate.dump(qubits.states)


.. note::

   Simulator.qubits.states internally calculates and copies values. For performance reasons, please make a copy of values.

.. code-block:: python

   sim.run(...)        # run a circuit.
   
   # Expected usage
   states = sim.states[:]     # copy all states
   for i in range(N) :
       v = states[i]
       ... use v to calculate something ...

   # Unexpected usage
   for i in range(N) :
       states = sim.states[i]  # accessing one by one, it's slow.
       ... use v to calculate something ...
   

Getting probability as array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Simulator.qubits.prob** returns array of probability, and is accessed like numpy arrays.  The same note for Simulator.qubits.states is applied for performance reasons.

**qgate.dump()** is also avaialble to dump probablity.

.. code-block:: python

   # getting whole state vector.
   v = sim.qubits.prob[:]

   # getting states for odd index-th elements.
   v = sim.qubits.prob[1::2]

   # dump probability
   qgate.dump(sim.prob)


Other examples
--------------

You can find other examples at `qgate github repository <https://github.com/shinmoirno/qgate>`_.

- `grover.py <https://github.com/shinmorino/qgate/tree/master/examples/grover.py>`_

  This exmple comes from `IBM Q Experience Tutorial. <https://www.ibm.com/developerworks/jp/cloud/library/cl-quantum-computing/index.html>`_.
  
- `quantum_fourier_transform.py <https://github.com/shinmorino/qgate/tree/master/examples/quantum_fourier_transform.py>`_

- `quantum_teleportation.py <https://github.com/shinmorino/qgate/tree/master/examples/quantum_teleportation.py>`_

  Above two examples are from examples in `OpenQASM article <https://github.com/Qiskit/openqasm/tree/master/spec-human>`_.
  
