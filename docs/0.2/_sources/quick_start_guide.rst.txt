Quick start guide
=================

This page covers followings:

#. `Simple example`_

#. `Code walkthrough`_
   
#. `Creating a quantum circuit`_
   
#. `Running a circuit on a simulator`_

#. `Accessing simulation results`_

#. `Accessing state vector`_

#. `Other examples`_


Simple example
^^^^^^^^^^^^^^

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


Here is an example of console output:

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
   

The package name is **qgate**, imported before using qgate.

Creating a quantum circuit
--------------------------

In Qgate, quantum circuits are defined as a list of :ref:`operators:Operators`.

.. code-block:: python
   :lineno-start: 3
		  
   from qgate.script import *
   
   # creating a qreg
   qreg = new_qreg()


The **qgate.script** package provides factory functions to create operators, qregs and refs.

Qreg is quantum register, which is the logical representation of qubit as deined in OpenQASM. Operators has qreg(s) to specify target and control qubits.

The function of **new_qreg()** returns one qreg instance.  To create a list of qregs, use **new_qregs(n_qregs)**.  Please see :ref:`qreg_and_reference:Quantum register and reference`.

.. code-block:: python
   :lineno-start: 8

   # creating a quantum circuit with one H gate
   circuit = [ H(qreg) ]

Here, one Hadamard gate is added to circuit.

For other available quantum gates in Qgate, please see :ref:`gate:Gate`.

.. code-block:: python
   :lineno-start: 11

   # creating reference as a placeholder to store a result. 
   ref = new_reference()

   # creating measure operation, and add it to the circuit.
   m = measure(ref, qreg)
   circuit.append(m)
   

For :ref:`measurement:Measurement`, a reference is used to refer a measured value.
With a reference and a qreg, measure operation is created and appended to circuit.

.. code-block:: python
   :lineno-start: 18
   
   # creating simulator instance, and run circuit.
   sim = qgate.simulator.cpu()
   sim.run(circuit)


To run simulations, simulator instance is created by using **qgate.simulator.cpu()**.  Other simulator factory methods of **qgate.simulator.py()** and **qgate.simulator.cuda()** are available as well.

Simulator instance has run() method, which accepts circuit as its parameter.  Simulator.run() returns after executing all operators in a given circuit.
   
.. code-block:: python
   :lineno-start: 22
		  
   # accesing simulation results
   value = sim.values.get(ref)
   print('Measured value: ' + str(value))

To get simulation results, **Simulator.values** property is used.  This property is a dictionary that holds measured values.

By passing the 'ref' object used at line 15, the measured result is retrieved by using sim.values.get(ref).

:ref:`measurement:Probability` operator is also available to get probability, :math:`Pr(Zero||\psi>)` on a specified qreg.  Also in this case, calculated probability is accessed in a similar way.
   
   
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


Running a circuit on a simulator
--------------------------------

Simulator instance is created by using **qgate.simulator.<runtime>()**, where runtime is py, cpu and cuda.

Qgate currently implements 3 versions of simulators, (1) python, (2) CPU(multicore), and (3) GPU(CUDA) versions, please see :ref:`simulator:Simulator` for details.

Simulator holds simulation results.  They are accessed from properties of **Simulator.values** and **Simulator.qubits**.

Accessing simulation results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Simulator.values** is a dictionary of values obtained during simulation.

In Qgate's quantum circuits, references are used to refer results obatained from measure and prob operations.

By calling Simulator.values.get(reflist), measurement results and prob values are obtained.  This method accepts one reference or a reference list.  If one reference is passed, one referred value is returned.  If a list of references is passed, a list of referred values is returned.

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
   qgate.dump(sim.qubits.states)


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
   qgate.dump(sim.qubits.prob)


Other examples
--------------

Please take a look at :ref:`sampling_example:Sampling example (Phase estimation)` as a more practical example.

You can find other examples at `qgate github repository <https://github.com/shinmorino/qgate>`_.

- `grover.py <https://github.com/shinmorino/qgate/tree/master/examples/grover.py>`_

  | This example is based on `IBM Q Experience Tutorial. <https://www.ibm.com/developerworks/jp/cloud/library/cl-quantum-computing/index.html>`_
  
- `quantum_fourier_transform.py <https://github.com/shinmorino/qgate/tree/master/examples/quantum_fourier_transform.py>`_

- `quantum_teleportation.py <https://github.com/shinmorino/qgate/tree/master/examples/quantum_teleportation.py>`_

  | Above two examples are from examples in `OpenQASM article <https://github.com/Qiskit/openqasm/tree/master/spec-human>`_.
