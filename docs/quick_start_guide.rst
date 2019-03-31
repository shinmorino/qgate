Quick start guide
=================

This page covers followings:

#. Simple example

#. Walkthrough
   
#. Building a circuit circuit

#. Running a circuit on a simulator

#. Showing simulation results


Simple example
--------------

The first example is to apply Hadmard gate to a quantum register and measure it.

The original source is **here**.

.. code-block:: python
   :linenos:
      
   # importing qgate packages
   import qgate
   from qgate.script import *
   
   # creating a qreg.
   qreg = new_qreg()

   # creating a quantum circuit with H gate.
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

Meaured values in this script are 0 or 1.


Code walkthrough
^^^^^^^^^^^^^^^^

In this section, we'll go through the above sample with detailed comments.

.. code-block:: python
   :lineno-start: 1
   
   # importing qgate packages
   import qgate
   from qgate.script import *
   

The package name is **qgate**.  Please import this before using qgate.

The **qgate.script** package provides factory functions for gates and other operators.

Quantum circuits are built on circuit, and circuit is  used as an input of simulators.
Circuit holds a sequence of operators.

.. code-block:: python
   :lineno-start: 5
		  
   # creating a qreg
   qreg = new_qreg()

Creating a qreg.

Qreg is quantum register as deined in OpenQASM.  It is a logical representation of qubit.

new_qreg() returns one qreg instance.  To create a list of qregs, use new_qregs(n_qregs).
   

.. code-block:: python
   :lineno-start: 8

   # creating a quantum circuit with H gate
   circuit = [ H(qreg) ]

In qgate, quantum circuits are defined as a sequence of operators, and python's list is used to store this sequence.
   In this example, Hadmard gate is added with a target bit specified by qreg.

Qgate implements various gates including controlled gates.  For the list of built-in gates, please visit **here**.

.. code-block:: python
   :lineno-start: 11

   # creating reference as a placeholder to store a result. 
   ref = new_reference()

   # creating measure operation, and add it to the circuit.
   m = measure(ref, qreg)
   circuit.append(m)
   

For measurement, reference is used as a placeholder of measured value.
With a created reference, measure operation is added to circuit.

.. code-block:: python
   :lineno-start: 18
   
   # creating simulator instance, and run circuit
   sim = qgate.simulator.cpu()
   sim.run(circuit)


To run simulations, simulator instance is created by using **qgate.simulator.cpu()**.

Simulator instance has run() method, which accepts circuit as its parameter.  Simulator.run() returns after executing all operators added to circuit.
   
.. code-block:: python
   :lineno-start: 22
		  
   # accesing simulation results
   value = sim.values.get(ref)
   print('Measured value: ' + str(value))

To get simulation result, Simulator.values property is used.  This property is a dictionary that maps obtained values to references.  Since 'ref' variable is used for measure operation at line 18, the measured value is retrieved by using sim.values.get(ref).  Here, reference is used as the key to access to the value.  The resulting value is 0 or 1.

There's Prob operator that calculates probability (<qreg|0><0|qreg>) on a specified qreg.  Also in this case, reference is used as a key to access values.  The resulting value type is float [0., 1.].


Opeartors
---------

In qgate, quantum circuits is defined as a sequence of operators in circuit.  Built-in operators are shown below.
  
  * Gate
    
  * Measurement

  * Probability
    
  * If clause
    
  * Reset
    
  * Barrier


Gate
^^^^

Table 1 shows the list of built-in gates.

================ ============================================================================
 Type            Gate
================ ============================================================================
 1 qubit gate     
   No parameter    ID, H, X, Y, Z, S, T
   1 parameter     Rx(theta), Ry(theta), Rz(theta), U1(_lambda), Expii(theta), Expia(theta)
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

To add single qubit gates, the syntax shown below is used.

Tokens surrounded by ``<>`` may appear 0- or 1-time according to gates to be declared.

.. code-block:: python

  <cntr(qregs).>GateType<(paramters)><.Adj>(qreg)

- Control bits

  ``cntr(qregs).`` specify control bits.  It appears only when controlled gates are decalared.
  A comma-seperated list of qregs, a list of qregs, or their mixture is accepted.

- GateType<(parameters)>

  GateType is the gate name, such as H, Rx and Expii.
  If a specified gate type does not have any parameter, ``(paramters)`` is omitted.

- <.Adj>

  Specifying a gate is adjoint of GateType.
  All gates except for Swap gate support adjoint.
  Gates such as H and X are hermite, so their adjoint is identical.  In these cases, .Adj is simply ignored.

- (qreg)

  Qreg instance as a target qubit(qreg).


Examples:

.. code-block:: python

   # Hadamard gate
   H(qreg0)

   # Controlled X gate (CX gate)
   cntr(qreg0).X(qreg1)

   # 2-control-bit X gate (Toffoli gate)
   cntr(qreg0, qreg1).X(qreg2)

   # Rx gate (1 parameter)
   Rx(0.)(qreg)

   # Adjoint of Rx gate
   Rx(0.).Adj(qreg)

   # adjoint of 3-bit-controlled U3 gate
   # control bits are given by a python list.
   cntrbits = [qreg0, qreg1, qreg2]  # creating a list of control bits
   cntr(cntrbits).U3(theta, phi, _lambda).Adj(qreg3)


Composed gate
+++++++++++++

Currently only 1 composed gate, Expi, is implemented.

The syntax to declare Expi gate is similar to other gates.  It allows to accept controll bits and supports adjoint.  But the operand is a list of pauli and ID gates.

.. code-block:: python

   <cntr(qregs).>GateType<(paramters)><.Adj>(gatelist)

Examples:

.. code-block:: python

   # exp(i * math.pi * X), identical to Rx(math.pi).
   Expi(math.pi)(X(qreg))

   # can have a sequence of pauli operators
   Expi(math.pi / 2.)(X(qreg0), Y(qreg1), Z(qreg2))
   
   # Can be a controlled gate
   cntr(qreg0).expi(math.pi)(Y(qreg1))
   
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

Circuit is defined by using python's list.  So sequences of gates and operators can be created programatically.  A nested list is also allowed.

.. code-block:: python

   # example of nexted list
   qregs = new_qregs(10)
   circuit = [
     [H(qreg) for qreg in qregs],  # creating a list with 10 H gates
     [X(qreg) for qreg in qregs]  # adding 10 X gates
   ]

   refs = new_references(10)
   # add 10 measure operators.
   circuit += [measure(ref, qreg) for ref, qreg in zip(refs, qregs)]


Measure operations
^^^^^^^^^^^^^^^^^^

Q gates implements 2 measure operations, (1) single-qubit measurement and (2) multi-qubit measurement.

Single qubit measurement
++++++++++++++++++++++++++++++++

Single qubit measuremenet is Z-based.  It's identical to measurement operation defined in OpenQASM.

Measure operation in qgate has a reference as the first parameter that refers to a measured value.  The second parameter is qreg, on which measurement is performed.

.. code-block:: python

   ref = new_reference()
   m = measure(ref, qreg)
   circuit.append(m)


Multi-qubit measurement
+++++++++++++++++++++++

Multi qubit measurement has a sequence of pauli gates to specify observable, which is an equivalent of Pauli measurement in Q#.
The first parameter is a reference, and it's the same as single qubit measurement.  And the second parameter is a sequence of pauli and identity gates .

.. code-block:: python

   ref = new_reference()
   gatelist = [X(qreg0), Y(qreg1)]
   m = measure(ref, gatelist)
   circuit.append(m)

   
Probability oprators
^^^^^^^^^^^^^^^^^^^^

Probablity opeartion calculates probablity for a specified qreg, <qreg|0><0|qreg>.

There're 2 probability operations, (1) single qubit probability and (2) multi-qubit probability.  It's very similar to measurement operations, but returns probability as floting number, [0., 1.).

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

If clause is for conditional execution of quantum circuits.

if_(ref, cond, clause)

The first argument, ref, is a reference or a list of references, The second parameter, cond, is a integer value or a function.  The third parameter, clause, is an operator or a list of operators.


cond as integer value
+++++++++++++++++++++

When ref is a reference, values referenced by refs are compared with the cond value.  If they're equal, clause is executed.

The paramter, ref, can be a list of references.  In thie case, ref is converted to an integer value accodring to the code shown below, and compared with the cond value.  This mimics OpenQASM if statement.

If ref is not measured, referenced value is 0.

.. code-block:: python
		
   v = 0
   for idx, ref in enumerate(refs):
       if get_value_from_ref(ref) == 1 :
           v |= 1 << idx

.. code-block:: python

   # if
   circuit.append( if_(ref, 1, X(qreg)) )

   # if
   refs = new_references(3)
   
   ... measure somewhere.

   # if values referred by refs[0] and refs[1] are 1,
   # and one referred by refs[2] is not 1,
   # X(qreg) is applied to qreg.
   circuit.append( if_(refs, 3, X(qreg)) )
   

cond as an function
+++++++++++++++++++

When ref is a reference, values referenced by refs are passed to the function specified in cond.  If the function returns True, clause is executed.

The paramter, ref, can be a list of references.  In thie case, unpacked values referred by refs are passed to the function.  If the function returns True, clause is executed.

If ref is not measured, referenced value is None.

.. code-block:: python

   # if with one reference
   # if a value referred by ref is 0, X(qreg) is executed.
   circuit.append( if_(ref, lambda x: return x == 0, [X(qreg)]) )

   # if with a list of references
   refs = new_references(3)
   
   ... measure somewhere.

   # if values referred by refs[0] and refs[1] are 1,
   # and one referred by refs[2] is 0,
   # X(qreg) is applied to qreg.
   circuit.append( if_(refs,
                       lambda v0, v1, v2 : return (v0 == 1) and (v1 == 1) and (v2 == 0),
		       X(qreg)) )
   
Reset
^^^^^

Reset operation resets qubits. Reset can accept a list of qregs.

A qreg should be measured before Reset operation.

If the qubit values is 1, X gate is applied.

The qubits is not measured, Reset will report an error.

.. code-block:: python

   # resetting qubit
   circuit.append(reset(qreg))

   # equivalent code.
   
   measure(ref, qreg)     # qreg is measured somewhere before reset().
   ...
   if_(ref, 1, X(qreg))
		
   
Barrier
^^^^^^^

Barrier operation works as barrier on quantum circuit optimization (to be implemented later versions).  Reset can accept a list of qregs.

.. code-block:: python

   # barrier, 1 qreg
   circuit.append(barrier(qreg))

   # barrier, 2 qregs
   circuit.append(barrier([qreg0, qreg1]))



Simulator
---------

Qgate currently has 3 versions of simulators, (1) python, (2) CPU(multicore), and (3) GPU(CUDA) versions.  Each runtime is created by using **qgate.simulator.py()**, **qgate.simulator.cpu()** or **qgate.siulator.cuda()** respectively.


Accessing simulation results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Properties of **Simulator.values** and **Simulator.qubits** are provided to access to simulation results.

Simulator.values is a dictionary of values obtarined by measure and prob operations.  These values are accessible by using references as its key.

Simulator.values.get() is used to get referred values, and it accepts one reference or a list of references.  If one reference is passed, one referred value is returned.  If a list of references is passed, a list of referred values is returned.

.. code-block:: python

   # getting one value associated with ref.
   v = sim.values.get(ref)

   # getting value list associated with reference list
   values = sim.values.get(refs)

Accessing state vector
^^^^^^^^^^^^^^^^^^^^^^

**Simulator.qubits.states** return copy of state vector, and is accessed like numpy arrays.

**qgate.dump()** is avialble to dump state vector.

.. code-block:: python

   # getting whole state vector.
   v = sim.qubits.states[:]

   # getting states for odd index-th elements.
   v = sim.qubits.states[1::2]

   # dump states
   qgate.dump(qubits.states)


Getting probability as array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Simulator.qubits.prob** returns array of probability, and is accessed like numpy arrays.

.. code-block:: python

   # getting whole state vector.
   v = sim.qubits.prob[:]

   # getting states for odd index-th elements.
   v = sim.quits.prob[1::2]

   # dump probability
   qgate.dump(sim.prob)
